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
        orbital_label: None,
        orbital_waiting: false,
        draft_banner: None,
        bonds_label: None,
        pseudo_generating: None,
        probe: None,
    }
}

pub(crate) fn model() -> ModelStatus {
    ModelStatus {
        teacher: "GpuKs",
        fitted: true,
        trusted: true,
        readiness: Some(sim::Readiness { answers: true, rule: None, atom: None, text: "strict mode answers here".into(), unlearned: None }),
        teach_offer: None,
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
        box_affected: 0,
        trust_floor: Some(70),
        fit_vs_held: Some(crate::sim::fit_vs_held(271, 273, 0, 2, 0, 0, 0)),
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
        class_loo: vec![sim::ClassLoo { class: "pair", samples: 273, force_rms: 0.009, force_max: 0.1, energy_max: 0.152, trusted: true, refusal: None }],
        last_audit_this_fit: true,
        solves: sim::SolveCounts::default(),
        trust_checks: Vec::new(),
        stuck_info: None,
        model_gen: 4,
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
        activity: Default::default(),
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
    // a Z that cannot be placed is refused, from a script or anywhere (final review app M3)
    for a in [Action::Element(26), Action::ElementIsotope(26, 56), Action::Element(0)] {
        let out = act(&mut st, &inp, a.clone());
        assert!(out.error.as_deref().is_some_and(|e| e.contains("only Z 1–18")), "{a:?}: {:?}", out.error);
        assert_eq!(st.element, 8, "{a:?}");
    }
    for words in [&["element", "26"][..], &["element", "200", "1"], &["element", "0"]] {
        assert!(Action::parse(words).is_err(), "{words:?}");
    }
    assert_eq!(Action::parse(&["element", "10", "22"]), Ok(Action::ElementIsotope(10, 22)));
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
    // a preset is loaded under the policy in force (M7: accepted species by default since Task 11; the preset file itself is all-electron)
    let want = Scene { ionic: st.ionic, ..crate::quality::preset("h2o", crate::quality::DEFAULT_H).unwrap() };
    assert_eq!((want.grid_n, want.grid_h, want.ionic), (95, 0.15, DEFAULT_IONIC));
    // the scene, then the container moved to the new cell's default (walls on by default)
    assert!(matches!(out.commands.as_slice(), [Command::SetScene(s), Command::Dyn(DynCommand::SetWalls(Some(w)))] if *s == want && (w.half_width - 5.55).abs() < 1e-12), "{:?}", out.commands);
    assert_eq!((st.source.preset(), st.grid_n, st.box_len), (Some("h2o"), 95, 14.0));
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
    assert!(out.commands.is_empty() && st.source.preset() == Some("he"));
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
    let out = act(&mut st, &inp, Action::LoadScene { sidecar: true });
    // the scene, then the container at the loaded cell's default (walls on by default)
    assert!(matches!(out.commands.as_slice(), [Command::SetScene(s), Command::Dyn(DynCommand::SetWalls(Some(_)))] if *s == scene), "{:?}", out.commands);
    let out = act(&mut st, &inp, Action::LoadModel);
    assert!(out.error.as_deref().is_some_and(|e| e.starts_with("no model file")));
    // estimated with a sidecar: the scene and its model
    std::fs::write(dir.join("t.model"), b"x").unwrap();
    st.backend = Backend::Estimated;
    let out = act(&mut st, &inp, Action::LoadScene { sidecar: true });
    assert!(matches!(out.commands.as_slice(), [Command::SetScene(_), Command::Dyn(DynCommand::SetWalls(_)), Command::Dyn(DynCommand::LoadModel(_))]), "{:?}", out.commands);
    // a replay's `scene` line: the file only, never its sidecar (final review app M8)
    let out = act(&mut st, &inp, Action::LoadScene { sidecar: false });
    assert!(matches!(out.commands.as_slice(), [Command::SetScene(_), Command::Dyn(DynCommand::SetWalls(_))]), "{:?}", out.commands);
    assert!(matches!(one_dyn(&act(&mut st, &inp, Action::LoadModel)), DynCommand::LoadModel(_)));
    assert!(matches!(one_dyn(&act(&mut st, &inp, Action::SaveModel)), DynCommand::SaveModel(p) if p.ends_with("t.model")));
    let _ = std::fs::remove_dir_all(&dir);
}

/// A model file that is only a header (what `surrogate::sidecar::read_provenance` reads): the magic, the header length
/// and a version-4 JSON header with this provenance.
fn header_only_model(name: &str, grid_n: usize, h: f64, sigma: f64, tol: f64, charge: i32, mult: &str, teacher: &str, ionic: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("atomic-ui-offer-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let header = format!(
        "{{\"format\":\"ASMODEL4\",\"provenance\":{{\"grid_n\":{grid_n},\"h\":{h},\"sigma\":{sigma},\"tol\":{tol:e},\"charge\":{charge},\"multiplicity\":{mult},\"teacher\":\"{teacher}\",\"masses\":[[1,2]],\"ionic\":\"{ionic}\"}}}}"
    );
    let mut bytes = b"ASMODEL4".to_vec();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

/// User request 2026-10-18: a model whose settings match the scene loads as it always did; one that differs opens the
/// offer card (no command), Apply sends the scene edits and then the load, Cancel sends nothing.
#[test]
fn a_model_at_other_settings_opens_the_card_and_apply_sends_the_edits_then_the_load() {
    let scene = Scene::preset("h2").unwrap(); // 49³ × 0.25, all-electron, charge 0
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let mut st = UiState::new("h2", &scene);
    (st.gpu_available, st.backend) = (true, Backend::Estimated);
    // matching: straight to the driver (the recorded D isotope does not matter)
    let same = header_only_model("same.model", 49, 0.25, 0.25, 1e-6, 0, "null", "Kohn-Sham (GPU)", "none");
    st.model_file = Some(same.clone());
    assert!(matches!(one_dyn(&act(&mut st, &inp, Action::LoadModel)), DynCommand::LoadModel(p) if *p == same));
    assert!(st.model_offer.is_none());
    // another grid, σ, charge and multiplicity: a card, nothing sent
    let other = header_only_model("other.model", 81, 0.15, 0.2, 1e-6, 1, "2", "Kohn-Sham (GPU)", "none");
    st.model_file = Some(other.clone());
    let out = act(&mut st, &inp, Action::LoadModel);
    assert!(out.commands.is_empty() && out.error.is_none(), "{:?}", out.commands);
    let o = st.model_offer.clone().expect("the card");
    let names: Vec<&str> = o.rows.iter().map(|r| r.name).collect();
    assert_eq!(names, ["grid", "σ", "charge", "multiplicity", "isotopes"]);
    assert_eq!(o.rows[0].versus(), "model: 81³ × h 0.15 Bohr · scene: 49³ × h 0.25 Bohr");
    assert_eq!(o.blocked(), None);
    // Cancel: gone, nothing sent, the grid fields untouched
    let out = act(&mut st, &inp, Action::CancelModelOffer);
    assert!(out.commands.is_empty() && st.model_offer.is_none());
    assert_eq!((st.grid_n, st.grid_h), (scene.grid_n, scene.grid_h));
    // Apply: the edits in order, the isotopes (the model's single isotope of H is the deuteron), then the load
    act(&mut st, &inp, Action::LoadModel);
    let out = act(&mut st, &inp, Action::ApplyModelSettings);
    assert!(out.error.is_none(), "{:?}", out.error);
    assert!(
        matches!(
            out.commands.as_slice(),
            [
                Command::SetGrid { n: 81, h },
                Command::SetSigma(Some(sg)),
                Command::SetCharge(1),
                Command::SetMultiplicity(Some(2)),
                Command::Dyn(DynCommand::SetIsotope { index: 0, a: 2 }),
                Command::Dyn(DynCommand::SetIsotope { index: 1, a: 2 }),
                Command::Dyn(DynCommand::LoadModel(p)),
            ] if *h == 0.15 && *sg == 0.2 && *p == other
        ),
        "{:?}",
        out.commands
    );
    assert!(st.model_offer.is_none());
    assert_eq!((st.grid_n, st.grid_h), (81, 0.15), "the panel's pending grid follows, as after Apply grid");
    // a second Apply with no card does nothing
    assert!(act(&mut st, &inp, Action::ApplyModelSettings).commands.is_empty());
    // an unreadable header is the driver's to report, as before
    let junk = std::env::temp_dir().join(format!("atomic-ui-offer-{}", std::process::id())).join("junk.model");
    std::fs::write(&junk, b"x").unwrap();
    st.model_file = Some(junk.clone());
    assert!(matches!(one_dyn(&act(&mut st, &inp, Action::LoadModel)), DynCommand::LoadModel(p) if *p == junk));
}

/// A setting the app cannot change (the teacher tolerance, the teacher, another build's recipe) is named with its reason and
/// Apply sends nothing but an error; the card stays, and draws.
#[test]
fn settings_that_cannot_be_applied_disable_apply() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let mut st = UiState::new("h2", &scene);
    (st.gpu_available, st.backend) = (true, Backend::Estimated);
    let p = header_only_model("blocked.model", 81, 0.15, 0.15, 1e-5, 0, "null", "Kohn-Sham (CPU)", "cores/0123456789abcdef/shared0/0/stop2x0");
    st.model_file = Some(p);
    assert!(act(&mut st, &inp, Action::LoadModel).commands.is_empty());
    let o = st.model_offer.clone().unwrap();
    let blocked: Vec<(&str, bool)> = o.rows.iter().map(|r| (r.name, r.blocked.is_some())).collect();
    assert_eq!(blocked, [("grid", false), ("teacher", true), ("teacher tolerance", true), ("ionic potential", true), ("isotopes", false)]);
    assert!(o.blocked().unwrap().contains("not a setting"), "{:?}", o.blocked());
    let out = act(&mut st, &inp, Action::ApplyModelSettings);
    assert!(out.commands.is_empty(), "{:?}", out.commands);
    assert!(out.error.as_deref().is_some_and(|e| e.starts_with("the model's settings cannot be applied")), "{:?}", out.error);
    assert!(st.model_offer.is_some(), "the card stays up to read");
    // the card draws over the view and, with no input, sends nothing
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let out = frame(&ctx, &mut st, &inp, 1400.0, vec![]);
    assert!(out.commands.is_empty());
    // while a pass runs, Apply is refused even for an applicable offer
    let p = header_only_model("pass.model", 81, 0.15, 0.15, 1e-6, 0, "null", "Kohn-Sham (GPU)", "none");
    st.model_file = Some(p);
    act(&mut st, &inp, Action::LoadModel);
    let mut snap = snapshot(&scene, Backend::Estimated, Some(model()));
    snap.dynamics.sampling = Some(sim::SamplingProgress { stage: "perturbations", done: 3, total: 40, skipped: 0, failed: 0, cancelled: false, last_energy: f64::NAN, finished: false, target: None });
    let busy = input(&scene, Some(&snap), &cam);
    let out = act(&mut st, &busy, Action::ApplyModelSettings);
    assert!(out.commands.is_empty() && out.error.as_deref().is_some_and(|e| e.contains("sampling pass")), "{:?} {:?}", out.commands, out.error);
    assert!(st.model_offer.is_some());
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
    // teach around the blockers: the old command path, recorded but not replayable (it solves)
    assert!(matches!(one_dyn(&act(&mut st, &inp, Action::TeachAroundBlockers)), DynCommand::StartTeaching));
    assert!(Action::TeachAroundBlockers.describe().is_none());
    assert_eq!(crate::replay::unreplayable_kind(&Action::TeachAroundBlockers), Some("teach-around-blockers"));
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

/// The teach-around offer: no button without teachable blockers, a label with the blockers, the
/// solves and their time at the model's solve rate, the stage worded with its blocker.
#[test]
fn the_teach_around_button_names_its_cost() {
    let mut m = model();
    assert_eq!(crate::ui::dock::model::teach_label(&m), None, "no offer");
    m.teach_offer = Some(sim::TeachOffer { blockers: Vec::new(), refused: vec![(7, "pair", 3.2, "the model cannot represent it")], refused_total: 1, solves: 0, notes: Vec::new(), prompt: None });
    assert_eq!(crate::ui::dock::model::teach_label(&m), None, "only refused blockers: nothing to start");
    m.teach_offer = Some(sim::TeachOffer { blockers: vec![(345, "pair", 12.0, 0.4), (438, "many-body", 10.6, 0.7)], refused: Vec::new(), refused_total: 0, solves: 20, notes: Vec::new(), prompt: None });
    m.solve_s = Some(6.0);
    assert_eq!(crate::ui::dock::model::teach_label(&m).as_deref(), Some("Teach around 2 blockers (≈20 solves, ≈2:00)"));
    m.teach_offer = Some(sim::TeachOffer { blockers: vec![(345, "pair", 12.0, 0.4)], refused: Vec::new(), refused_total: 0, solves: 10, notes: Vec::new(), prompt: None });
    m.solve_s = None;
    assert_eq!(crate::ui::dock::model::teach_label(&m).as_deref(), Some("Teach around 1 blocker (≈10 solves)"));
    let p = sim::SamplingProgress { done: 1, total: 10, stage: "teaching", last_energy: f64::NAN, cancelled: false, skipped: 0, failed: 0, finished: false, target: Some(345) };
    assert_eq!(crate::ui::activity::words::pass_stage(&p), "teaching around #345");
}

/// The live case's prompt: a teach offer that carries a prompt (consecutive class-rule fallbacks
/// blocked far away) puts "stuck on DFT: …" on the status chip; none without the prompt or while
/// a pass runs.
#[test]
fn a_teach_prompt_reaches_the_status_chip() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let st = UiState::new("h2", &scene);
    let mut m = model();
    m.teach_offer = Some(sim::TeachOffer { blockers: vec![(251, "many-body", 1.0, 0.4)], refused: Vec::new(), refused_total: 0, solves: 10, notes: Vec::new(), prompt: None });
    let snap = snapshot(&scene, Backend::Estimated, Some(m.clone()));
    assert!(crate::ui::overlays::chip_notes(&st, &input(&scene, Some(&snap), &cam)).iter().all(|(_, t)| !t.starts_with("stuck on DFT")));
    m.teach_offer.as_mut().unwrap().prompt = Some("the last 5 steps fell back because three-or-more-atom environments are untrusted everywhere".into());
    let snap = snapshot(&scene, Backend::Estimated, Some(m));
    let notes = crate::ui::overlays::chip_notes(&st, &input(&scene, Some(&snap), &cam));
    assert!(notes.iter().any(|(_, t)| t.starts_with("stuck on DFT: the last 5 steps fell back")), "{notes:?}");
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

/// Atom trails (user request 2026-10-17): off by default at 100 fs and 70 %, found by Ctrl K, and
/// display only — drawing the sliders sends nothing to the worker.
#[test]
fn trail_controls_are_display_only_and_searchable() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    assert!(!st.show_trails);
    assert_eq!((st.trail_fs, st.trail_opacity), (100.0, 0.7));
    let r = search::results("trails", 3);
    assert_eq!((r[0].name, r[0].anchor), ("Trails", "m-display"));
    assert!(search::results("decay", 50).iter().any(|e| e.name == "Trails"));
    let inp = input(&scene, None, &cam);
    st.show_trails = true;
    egui::Popup::open_id(&ctx, menus::popup_id(menus::MenuId::View));
    for w in [1400.0, 1000.0] {
        for _ in 0..2 {
            assert!(frame(&ctx, &mut st, &inp, w, vec![]).commands.is_empty());
        }
    }
    egui::Popup::close_all(&ctx);
    assert!(st.show_trails && st.trail_fs == 100.0 && st.trail_opacity == 0.7, "drawing changes nothing");
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
        target: None,
    });
    // a snapshot whose activity carries events of each origin and full training views, so the
    // Activity tab draws its log, its checklist and its plots
    let mut rich = snapshot(&scene, Backend::Estimated, Some(model()));
    rich.activity.events = activity_tab::every_kind_of_event();
    rich.activity.views = Some(std::sync::Arc::new(activity::fixture_views()));
    st.activity.absorb(&rich);
    let snaps = [
        None,
        Some(snapshot(&scene, Backend::Gpu, None)),
        Some(snapshot(&scene, Backend::Estimated, Some(m))),
        Some(sampling),
        Some(rich),
    ];
    for snap in &snaps {
        for (w, sel) in [(1400.0, None), (1000.0, Some(0)), (1400.0, Some(2))] {
            let inp = PanelInput {
                selected: sel,
                playing: sel.is_some(),
                density_label: Some("density from the last DFT step".into()),
                orbital_label: Some("orbitals from t = 1.00 fs (last DFT solve). The estimated solver has no electrons".into()),
                last_error: sel.map(|_| "an error"),
                ..input(&scene, snap.as_ref(), &cam)
            };
            for tab in DockTab::ALL {
                st.open_tab(tab);
                for _ in 0..2 {
                    let out = frame(&ctx, &mut st, &inp, w, vec![]);
                    assert!(out.commands.is_empty() && !out.toggle_play && !out.deselect && out.error.is_none(), "{tab:?} at {w}");
                }
            }
            (st.dock_open, st.left_open) = (false, false);
            frame(&ctx, &mut st, &inp, w, vec![]);
            (st.dock_open, st.left_open) = (true, true);
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
    st.open_tab(DockTab::Model);
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

/// A scene loaded from a file is the chip's source (final review app I1/I3): the chip names the
/// file, the file's spacing becomes the quality later presets load at, no preset is highlighted so
/// a click on `h2` loads it, and Reset reloads the file — even after the path field was edited.
#[test]
fn a_loaded_scene_file_is_the_chips_source_and_reset_reloads_it() {
    let dir = std::env::temp_dir().join(format!("atomic-ui-source-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("foo.scene").display().to_string();
    let file = Scene::preset("lih").unwrap(); // the file as written: 57³ × 0.25
    std::fs::write(&path, file.to_text()).unwrap();
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let snap = snapshot(&scene, Backend::Gpu, None);
    let inp = input(&scene, Some(&snap), &cam);
    let mut st = UiState::new("h2", &scene);
    st.scene_path = path.clone();
    let out = act(&mut st, &inp, Action::LoadScene { sidecar: true });
    assert!(matches!(out.commands.first(), Some(Command::SetScene(s)) if *s == file), "{:?}", out.commands);
    assert_eq!(st.source, super::SceneSource::File(path.clone()));
    assert_eq!((st.source.label().as_str(), st.source.preset()), ("foo.scene", None));
    assert_eq!((st.quality_h, st.quality_slider, st.grid_n), (0.25, 0.25, 57), "the file's spacing is the quality");
    // no preset is current: h2 is clickable and loads at the file's spacing
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    egui::Popup::open_id(&ctx, egui::Id::new("scene-chip"));
    let outs = click_widget(&ctx, &mut st, &inp, "h2");
    let sent: Vec<&Command> = outs.iter().flat_map(|o| &o.commands).collect();
    assert!(matches!(sent.first(), Some(Command::SetScene(s)) if s.nuclei.len() == 2 && (s.grid_n, s.grid_h) == (49, 0.25)), "{sent:?}");
    assert_eq!(st.source.preset(), Some("h2"));
    // back on the file; the path field edited since: Reset reloads the file, not the field
    let _ = act(&mut st, &inp, Action::LoadScene { sidecar: true });
    st.scene_path = "elsewhere.scene".into();
    let outs = click_widget(&ctx, &mut st, &inp, "reset");
    egui::Popup::close_all(&ctx);
    let sent: Vec<&Command> = outs.iter().flat_map(|o| &o.commands).collect();
    assert!(matches!(sent.first(), Some(Command::SetScene(s)) if *s == file), "{sent:?}");
    assert_eq!(st.scene_path, path);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The scene chip's Presets popup lists `empty` first, and clicking it sends the same
/// `SetScene` (→ the worker's `ReplaceScene`) as every other preset, adopting its grid.
#[test]
fn the_presets_popup_lists_empty_first_and_sends_its_scene() {
    assert_eq!(Scene::PRESETS, &["empty", "h", "h2", "he", "lih", "h2o", "h2s", "hcl", "ph3", "sih4"]);
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
    let want = Scene { ionic: st.ionic, ..crate::quality::preset("empty", crate::quality::DEFAULT_H).unwrap() };
    let sent: Vec<&Command> = outs.iter().flat_map(|o| &o.commands).collect();
    assert!(matches!(sent.as_slice(), [Command::SetScene(s), Command::Dyn(DynCommand::SetWalls(Some(_)))] if *s == want), "{sent:?}");
    assert_eq!((st.source.preset(), st.grid_n), (Some("empty"), 81), "the empty preset at the default quality");
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
    st.open_tab(DockTab::Model);
    let inp = input(&scene, Some(&est), &cam);
    for _ in 0..3 {
        assert!(frame(&ctx, &mut st, &inp, 1400.0, vec![]).commands.is_empty());
    }
    // the sampling button is there but disabled: a click sends nothing
    let outs = click_widget(&ctx, &mut st, &inp, "sample");
    assert!(outs.iter().all(|o| o.commands.is_empty()), "an empty scene cannot be sampled");
    for snap in [None, Some(&gpu), Some(&est)] {
        for tab in DockTab::ALL {
            st.open_tab(tab);
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
        target: None,
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
    pass.dynamics.sampling = Some(sim::SamplingProgress { done: 1, total: 9, stage: "perturbations", last_energy: -1.1, cancelled: false, skipped: 0, failed: 0, finished: false, target: None });
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

#[test]
fn show_angles_action_round_trips_and_defaults_on() {
    let scene = Scene::preset("h2o").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let mut st = UiState::new("h2o", &scene);
    assert!(st.show_angles, "bond angles are on by default (spec §8 row 13)");
    for b in [true, false] {
        let text = Action::ShowAngles(b).describe().unwrap();
        assert_eq!(text, format!("angles {b}"));
        let words: Vec<&str> = text.split(' ').collect();
        assert_eq!(Action::parse(&words), Ok(Action::ShowAngles(b)));
    }
    assert!(act(&mut st, &inp, Action::ShowAngles(false)).commands.is_empty(), "display only: no command");
    assert!(!st.show_angles);
    act(&mut st, &inp, Action::ShowAngles(true));
    assert!(st.show_angles);
    assert!(search::results("angle", 3).iter().any(|e| e.name == "Bond angles"), "Ctrl K finds the toggle");
}

#[test]
fn angle_note_names_the_surface() {
    assert_eq!(angle_surface_note(&[8, 1], [8, 1, 1]), "pseudopotential surface");
    assert!(angle_surface_note(&[8], [8, 1, 1]).contains("smeared"));
    assert!(angle_surface_note(&[], [8, 1, 1]).contains("smeared"));
}

/// Plan review Q10: a bond point inside a pseudised atom's r_c gets a valence-density kind. O at the
/// origin with r_c,max = 1.6, H at 1.81 Bohr: the bond point at s = 0.75 (1.36 Bohr from O) is inside;
/// at s = 0.95 (1.72 Bohr) it is not, and nothing is inside when nothing is pseudised.
#[test]
fn valence_density_kind_inside_an_r_c_sphere() {
    let pos = [[0.0; 3], [1.81, 0.0, 0.0]];
    let o_only = |z: u8| (z == 8).then_some(1.6);
    assert!(valence_density_kind(&pos, &[8, 1], 0, 1, 0.75, o_only));
    assert!(!valence_density_kind(&pos, &[8, 1], 0, 1, 0.95, o_only));
    assert!(!valence_density_kind(&pos, &[8, 1], 0, 1, 0.75, |_| None), "nothing pseudised: the all-electron kind");
    assert!(valence_density_kind(&pos, &[8, 1], 1, 0, 0.25, o_only), "the same point seen from the other end");
    assert!(VALENCE_KIND_NOTE.starts_with("valence-density kind"));
}

/// Clicked as a user would: the toolbar's Relax (Stop while relaxing), Physics ▸ Relax, the
/// Selection tab's ²H, View ▸ bonds, and the periodic table's cells (O places, K is greyed and
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
    // View ▸ Display aids ▸ show what training is solving: on by default, display only (no command,
    // no recorded action), and found by Ctrl K
    assert!(st.show_ghost);
    let outs = in_menu(&mut st, &inp, menus::MenuId::View, "show-ghost");
    assert!(outs.iter().all(|o| o.commands.is_empty() && o.actions.is_empty()), "display only");
    assert!(!st.show_ghost);
    in_menu(&mut st, &inp, menus::MenuId::View, "show-ghost");
    assert!(st.show_ghost);
    assert_eq!(search::results("training is solving", 3)[0].name, "Show what training is solving");
    assert_eq!(search::results("ghost", 1)[0].name, "Show what training is solving");
    assert!(matches!(search::results("what training", 1)[0].place, search::Place::Menu(menus::MenuId::View)));
    st.open_tab(DockTab::Selection);
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
    in_table(&mut st, "ptable-19");
    assert_eq!(st.element, 8, "K is greyed: needs pseudopotentials");
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
        bonds: bonds.iter().map(|&(i, j)| sim::Bond { i, j, rho_bcp: 0.18, enhancement: 1.3, s: 0.6, route: sim::Route::Deformation, multi_centre: false, laplacian_bcp: -0.8, laplacian_noise: 0.02 }).collect(),
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
        laplacian: None,
        laplacian_noise: None,
    };
    assert_eq!(
        pair_line(&scene, &r(0, 1, sim::Verdict::Bonded { multi_centre: false }, Some(sim::Route::Deformation))),
        "O–H (#0–#1): bonded — ρ_b 0.180 e/Bohr³, 1.30× the free atoms' density there, deformation route"
    );
    // with the Laplacian: the strength clause in plain words (user request, M6)
    let with = |lap: f64| sim::PairReport { laplacian: Some(lap), laplacian_noise: Some(0.05), ..r(0, 1, sim::Verdict::Bonded { multi_centre: false }, Some(sim::Route::Saddle)) };
    assert!(pair_line(&scene, &with(-1.2)).ends_with("saddle route; Laplacian -1.200 e/Bohr⁵: covalent (shared)"), "{}", pair_line(&scene, &with(-1.2)));
    assert!(pair_line(&scene, &with(0.3)).ends_with("Laplacian +0.300 e/Bohr⁵: weak / stretched contact (closed-shell)"));
    assert!(pair_line(&scene, &with(0.01)).ends_with(": not resolved on this grid (the Laplacian is within its noise)"));
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
    st.open_tab(DockTab::Selection);
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
        st.open_tab(tab);
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

/// The trust-check list of an untrusted fit whose first unmet condition is `key` (the box
/// fixtures carry the list `box_blocks_trust` reads; plan review S15).
fn checks_failing(key: sim::activity::TrustKey) -> Vec<sim::activity::TrustCheck> {
    use sim::activity::{TrustCheck, TrustKey as K};
    let audit = if key == K::AuditForces { K::AuditForces } else { K::Audit };
    [K::Samples, K::ResidualDofE, K::ResidualDofF, K::EnergyResidual, K::ForceResidual, K::ForceNoise, audit]
        .into_iter()
        .map(|k| TrustCheck { key: k, met: k != key, value: 0.0, limit: 1.0 })
        .collect()
}

/// diag-3h (a user's H₂ + H at the Accurate grid's 12 Bohr cell, refused as "max training energy
/// residual 2.2e-3" with 130 of 145 samples box-affected): the trust readout names the box when
/// box-affected samples are still in the fit because the clean ones are below the trust floor,
/// and only then; the advice suggests a cell from the scene's extent and the box rule
/// (123³ × 0.15 here, next to the 121³ measured trusted), and says when the memory guard or the
/// Fix (a), 2026-10-18 (the user's water session: "trusted" while every step fell back): a model
/// that passes the global trust rule but that strict mode refuses reads, amber, "a class untrusted"
/// (rule 4b: the class is untrusted everywhere, round 1 SHOULD-5) or "trusted, not here" (a
/// geometry refusal), with the rule's words, in the chip, the Model tab and the Activity checklist
/// (a "Strict mode answers at this geometry" line marked "this is why"); a scene whose own pass is
/// not in the fit reads "not this scene"; one where strict mode answers stays "yes".
#[test]
fn a_trusted_model_that_strict_mode_refuses_here_says_so() {
    let (scene, st) = (Scene::preset("h2").unwrap(), UiState::new("h2", &Scene::preset("h2").unwrap()));
    let refused = sim::Readiness {
        answers: false,
        rule: Some(sim::activity::TeachRule::Class(sim::activity::EnvClass::ManyBody)),
        atom: None,
        text: "this kind of environment isn't learned well enough yet (many-body: LOO force error 12.0× the bound, RMS 0.10×, 587 samples) — worst leave-one-out sample #345".into(),
        unlearned: None,
    };
    let m = ModelStatus { readiness: Some(refused.clone()), quarantined: 0, ..model() };
    let t = trust(Some(&m), &scene, &st);
    // a class refusal is about the class everywhere, not this geometry (round 1, SHOULD-5)
    assert_eq!((t.kind, t.kind.short(), t.colour), (TrustKind::ClassUntrusted, "a class untrusted, see Model", AMBER));
    assert!(t.text.starts_with("yes overall, but this kind of environment isn't learned well enough yet (many-body"), "{}", t.text);
    // a geometry refusal reads "not here"
    let here = sim::Readiness { rule: Some(sim::activity::TeachRule::Grade), text: "extrapolating: grade γ = 3.1 > γ_max = 2".into(), ..refused.clone() };
    let t = trust(Some(&ModelStatus { readiness: Some(here), quarantined: 0, ..model() }), &scene, &st);
    assert_eq!((t.kind, t.kind.short()), (TrustKind::NotHere, "trusted, not here"));
    assert!(t.text.starts_with("not here — extrapolating"), "{}", t.text);
    let lines = activity::trust::lines(&m, &scene, &st);
    let here = lines.iter().find(|l| l.text.starts_with("Strict mode answers at this geometry")).expect("the readiness line");
    assert!(!here.met && here.why && here.text.contains("worst leave-one-out sample #345"), "{here:?}");
    // the scene's own pass is not in the fit: "not this scene", even where the policy answers
    let unlearned = sim::Readiness { answers: true, rule: None, atom: None, text: "strict mode answers here".into(), unlearned: Some("this scene is not learned — 70 of this pass's 70 samples are NOT in the fit".into()) };
    let t = trust(Some(&ModelStatus { readiness: Some(unlearned), quarantined: 0, ..model() }), &scene, &st);
    assert_eq!(t.kind, TrustKind::NotHere);
    assert!(t.text.starts_with("not this scene — this scene is not learned"), "{}", t.text);
    // strict mode answers: plain "yes"
    let t = trust(Some(&ModelStatus { quarantined: 0, ..model() }), &scene, &st);
    assert_eq!((t.kind, t.text.as_str()), (TrustKind::Trusted, "yes"));
    let lines = activity::trust::lines(&ModelStatus { quarantined: 0, ..model() }, &scene, &st);
    assert!(lines.iter().any(|l| l.met && l.text == "Strict mode answers at this geometry"));
}

/// grid limit would refuse it.
#[test]
fn the_trust_readout_names_the_box_when_it_keeps_the_model_untrusted() {
    let reason = "max training energy residual 2.2e-3 > 1.0e-3 Ha/atom";
    let m = ModelStatus {
        trusted: false,
        untrusted_reason: Some(reason.into()),
        samples: 145,
        quarantined: 0,
        box_affected: 130,
        box_excluded: 0,
        trust_floor: Some(59),
        trust_checks: checks_failing(sim::activity::TrustKey::EnergyResidual),
        ..model()
    };
    let (scene0, st0) = (Scene::preset("h2").unwrap(), UiState::new("h2", &Scene::preset("h2").unwrap()));
    let t = trust(Some(&m), &scene0, &st0);
    assert_eq!(t.kind, TrustKind::Untrusted);
    assert!(t.text.starts_with("no — Samples clear of the box walls — 130 in the fit touch them (145 held)"), "{}", t.text);
    assert_eq!(t.detail.as_deref(), Some(reason), "the fit's own check stays on hover");
    assert_eq!(t.tip, Some(BOX_TRUST_TIP));
    // enough clean samples (the exclusion will engage at the next refit) or none in the fit: the
    // plain refusal
    for other in [
        ModelStatus { box_affected: 3, ..m.clone() },
        ModelStatus { box_excluded: 130, ..m.clone() },
        ModelStatus { trusted: true, untrusted_reason: None, ..m.clone() },
    ] {
        let t = trust(Some(&other), &scene0, &st0);
        assert!(!t.text.contains("box walls"), "{}", t.text);
    }
    // a count refusal (a pass cancelled early) with the same box samples: Sample more is the
    // cure, so the plain refusal, not the box (M6 final re-review)
    use sim::activity::TrustKey as K;
    for (count, key) in [("n_samples 40 < q_prime 59", K::Samples), ("residual_dof_e 20.0 < 30", K::ResidualDofE), ("residual_dof_f 12.5 < 30", K::ResidualDofF)] {
        let other = ModelStatus { untrusted_reason: Some(count.into()), trust_checks: checks_failing(key), ..m.clone() };
        let t = trust(Some(&other), &scene0, &st0);
        assert_eq!(t.kind, TrustKind::Untrusted);
        assert!(!t.text.contains("box walls") && t.text.starts_with("no — Enough "), "{}", t.text);
        assert_eq!(t.detail.as_deref(), Some(count), "the count refusal's own text on hover");
        assert!(crate::sim::box_blocks_trust(&other).is_none());
    }
    // the advice: the scene's extent (x = 3.725) + d_box(0.15) = 5.35 Bohr on each side
    let mut scene = crate::quality::preset("h2", 0.15).unwrap();
    assert_eq!((scene.grid_n, scene.grid_h), (81, 0.15));
    let mut third = scene.nuclei[0].clone();
    third.pos = [3.725, 0.0, 0.0];
    scene.nuclei.push(third);
    let a = box_advice_with(&scene, 0, u64::MAX);
    assert!(a.starts_with("Enlarge the cell (Scene ▸ Grid: 123³ at 0.15 Bohr, a 18.3 Bohr box — every nucleus ≥ 5.3 Bohr from a face"), "{a}");
    assert!(!a.contains("refuse") && !a.contains("beyond"), "{a}");
    assert!(box_advice_with(&scene, 2, 1).contains("The GPU memory guard would refuse that cell"));
    // already inside that box (the samples reached farther than the scene does now): grow by
    // d_box on every side; past the largest grid: say so
    let mut far = scene.clone();
    far.nuclei[2].pos = [20.0, 0.0, 0.0];
    assert!(box_advice_with(&far, 0, u64::MAX).contains("beyond the largest grid (201³)"));
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


// ---- the orbital viewer (plan Task 7) ----

mod orbital_viewer {
    use super::super::dock::orbitals::{self, Levels, RowView};
    use super::super::overlays::legend_of;
    use super::*;
    use crate::viewer::{OrbitalShown, ORBITALS_CPU_TEACHER, ORBITALS_NEED_DFT, ORBITALS_NO_ELECTRONS, SURROGATE_DENSITY_FADE};
    use sim::{LoneAtomLabels, OrbitalDisplay, OrbitalInfo, OrbitalLabel, OrbitalRequest, SideKind, Spin, SpinSummary, UnoccupiedStatus};

    fn row(spin: Spin, index: usize, track: u32, slot: u8, e: f64, occ: f64, label: OrbitalLabel) -> OrbitalInfo {
        OrbitalInfo {
            spin, index, track, slot, energy: e, eps_range: (e, e), occupation: occ, label, cluster: None, rotated: false,
            grid_split: None, box_state: false, smeared_1s: false, residual: None, not_converged: false,
        }
    }

    fn idx(k: usize) -> OrbitalLabel {
        OrbitalLabel::Index(k)
    }

    /// A payload for `what` over `rows` (no field: the chrome reads rows, status and notes).
    fn payload(rows: Vec<OrbitalInfo>, what: OrbitalRequest) -> OrbitalDisplay {
        OrbitalDisplay {
            cell: sim::Cell { n: 33, h: 0.25 },
            origin: [-4.0; 3],
            positions: vec![[0.0; 3]],
            rows,
            field: None,
            spin: SpinSummary::default(),
            unoccupied: None,
            notes: Vec::new(),
            what,
            lone_atom: LoneAtomLabels::NotLoneAtom,
        }
    }

    /// An H₂-like list: σ_g in both channels, σ_u unoccupied, with the track numbers 1, 2, 3.
    fn h2_rows() -> Vec<OrbitalInfo> {
        vec![
            row(Spin::Up, 0, 1, 0, -0.5, 1.0, idx(1)),
            row(Spin::Up, 1, 3, 1, 0.02, 0.0, idx(2)),
            row(Spin::Down, 0, 2, 0, -0.5, 1.0, idx(1)),
        ]
    }

    fn with_payload(scene: &Scene, backend: Backend, o: OrbitalDisplay) -> Snapshot {
        let mut s = snapshot(scene, backend, None);
        s.dynamics.orbitals = Some(o);
        s
    }

    fn orbital_view_commands(outs: &[PanelOutput]) -> Vec<sim::OrbitalViewRequest> {
        outs.iter()
            .flat_map(|o| &o.commands)
            .filter_map(|c| match c {
                Command::OrbitalView(r) => Some(r.clone()),
                _ => None,
            })
            .collect()
    }

    /// Clicking Orbitals in View ▸ Density ▸ show sends the orbital view for the HOMO (a placeholder
    /// track the producer resolves, until a payload says which) with no unoccupied orbitals yet,
    /// and opens the dock on the Orbitals tab; clicking Density switches the view off.
    #[test]
    fn show_selector_sends_the_request() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let snap = snapshot(&scene, Backend::Gpu, None);
        let inp = input(&scene, Some(&snap), &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        st.dock_open = false;
        egui::Popup::open_id(&ctx, menus::popup_id(menus::MenuId::View));
        let outs = click_widget(&ctx, &mut st, &inp, "menu-show-orbitals");
        egui::Popup::close_all(&ctx);
        let sent = orbital_view_commands(&outs);
        assert_eq!(sent.len(), 1, "one request");
        assert_eq!(sent[0].what, OrbitalRequest::Orbital { spin: Spin::Up, track: crate::ui::HOMO_TRACK, squared: false });
        assert_eq!(sent[0].unoccupied, 0, "the eigensolve waits for the first payload (Viewer::drive_orbital_view)");
        assert!(st.unocc_deferred && st.show == ShowMode::Orbitals);
        assert!(st.dock_open && st.dock_tab == DockTab::Orbitals, "the Orbitals tab opens");
        // the other modes
        for (mode, want) in [
            (ShowMode::Coloured, OrbitalRequest::AllColoured { include: None, physical: false, look: Default::default() }),
            (ShowMode::Spin, OrbitalRequest::SpinDensity),
            (ShowMode::Density, OrbitalRequest::Off),
        ] {
            let out = act(&mut st, &inp, Action::Show(mode));
            match out.commands.as_slice() {
                [Command::OrbitalView(r)] => assert_eq!(r.what, want, "{mode:?}"),
                c => panic!("{mode:?}: {c:?}"),
            }
        }
        assert!(!st.unocc_deferred);
        // with a payload on screen the HOMO is a real track: the σ_g of the ↑ channel (the first of
        // the equal pair at −0.5 Ha)
        let with = with_payload(&scene, Backend::Gpu, payload(h2_rows(), OrbitalRequest::List));
        let inp = input(&scene, Some(&with), &cam);
        let out = act(&mut st, &inp, Action::Show(ShowMode::Orbitals));
        let r = orbital_view_commands(&[out]);
        assert!(matches!(r[0].what, OrbitalRequest::Orbital { spin: Spin::Up, track: 1, squared: false }), "{:?}", r[0].what);
    }

    /// The tab's controls: a row picks by track, "show unoccupied" clamps to 8 and is sent when set,
    /// the include boxes start from every occupied orbital, the blend is a render setting (no
    /// request), the colour scale is a request, dots draw an orbital as ψ̃², and a script can say
    /// all of it.
    #[test]
    fn orbital_actions_send_the_view_and_round_trip_as_script_words() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let snap = with_payload(&scene, Backend::Gpu, payload(h2_rows(), OrbitalRequest::List));
        let inp = input(&scene, Some(&snap), &cam);
        let mut st = UiState::new("h2", &scene);
        let one = |out: PanelOutput| -> sim::OrbitalViewRequest {
            match out.commands.as_slice() {
                [Command::OrbitalView(r)] => r.clone(),
                c => panic!("{c:?}"),
            }
        };
        let r = one(act(&mut st, &inp, Action::PickOrbital { spin: Spin::Down, track: 2 }));
        assert_eq!((st.show, st.orbital_pick), (ShowMode::Orbitals, Some((Spin::Down, 2))));
        assert_eq!(r.what, OrbitalRequest::Orbital { spin: Spin::Down, track: 2, squared: false });
        let r = one(act(&mut st, &inp, Action::OrbitalPhase(false)));
        assert!(matches!(r.what, OrbitalRequest::Orbital { squared: true, .. }), "|ψ|²");
        act(&mut st, &inp, Action::OrbitalPhase(true));
        let r = one(act(&mut st, &inp, Action::VolumeMode(crate::ui::VOLUME_DOTS)));
        assert!(matches!(r.what, OrbitalRequest::Orbital { squared: true, .. }), "dots draw an orbital as its square");
        let r = one(act(&mut st, &inp, Action::VolumeMode(crate::ui::VOLUME_VOLUME)));
        assert!(matches!(r.what, OrbitalRequest::Orbital { squared: false, .. }));
        assert_eq!(one(act(&mut st, &inp, Action::Unoccupied(12))).unoccupied, 8, "clamped to 0–8");
        assert!(!st.unocc_deferred);
        // all coloured: the include boxes start from the occupied tracks
        st.show = ShowMode::Coloured;
        let r = one(act(&mut st, &inp, Action::Include { track: 1, on: false }));
        assert_eq!(r.what, OrbitalRequest::AllColoured { include: Some(vec![2]), physical: false, look: Default::default() });
        let r = one(act(&mut st, &inp, Action::Include { track: 3, on: true }));
        assert_eq!(r.what, OrbitalRequest::AllColoured { include: Some(vec![2, 3]), physical: false, look: Default::default() });
        assert!(act(&mut st, &inp, Action::ColourBlend { additive: false }).commands.is_empty(), "the blend is a render setting");
        assert!(!st.colour_additive);
        let r = one(act(&mut st, &inp, Action::ColourScale { physical: true }));
        assert!(matches!(r.what, OrbitalRequest::AllColoured { physical: true, .. }), "{:?}", r.what);
        let r = one(act(&mut st, &inp, Action::IncludeAllOccupied));
        assert_eq!(r.what, OrbitalRequest::AllColoured { include: None, physical: true, look: Default::default() });
        // the words
        for a in [
            Action::Show(ShowMode::Coloured),
            Action::PickOrbital { spin: Spin::Down, track: 7 },
            Action::OrbitalPhase(false),
            Action::Unoccupied(3),
            Action::Include { track: 4, on: false },
            Action::IncludeAllOccupied,
            Action::ColourBlend { additive: false },
            Action::ColourScale { physical: true },
            Action::ColourLook(sim::ColourLook { enclose: Some(75), shell: false, hide_core: true }),
            Action::ColourLook(sim::ColourLook { enclose: Some(90), shell: true, hide_core: false }),
        ] {
            let text = a.describe().expect("a script form");
            let words: Vec<&str> = text.split(' ').collect();
            assert_eq!(Action::parse(&words).unwrap(), a, "{text}");
        }
        assert!(Action::parse(&["show", "plasma"]).unwrap_err().contains("density, orbitals, colour or spin"));
    }

    /// `run_settings` carries the orbital view, so a solver switch (a new worker) keeps it.
    #[test]
    fn run_settings_carry_the_orbital_view() {
        let scene = Scene::preset("h2").unwrap();
        let mut st = UiState::new("h2", &scene);
        assert!(!st.run_settings().iter().any(|c| matches!(c, DynCommand::SetOrbitalView(_))), "off by default");
        st.show = ShowMode::Spin;
        let v = st.run_settings();
        assert!(v.iter().any(|c| matches!(c, DynCommand::SetOrbitalView(r) if r.what == OrbitalRequest::SpinDensity)), "{v:?}");
        st.show = ShowMode::Orbitals;
        st.orbital_pick = Some((Spin::Down, 5));
        let v = st.run_settings();
        assert!(v.iter().any(|c| matches!(c, DynCommand::SetOrbitalView(r) if r.what == OrbitalRequest::Orbital { spin: Spin::Down, track: 5, squared: false })), "{v:?}");
    }

    /// M7 Task 9 (task-9 carry item 6; Task 6 review A.6): a lone pseudo-O's list — the minority 2p shared over three
    /// degenerate levels (f = 1/3 each), a gap witness at f = 0 above — shows the fractions, and no HOMO-LUMO gap is
    /// quoted when a shell is partly filled (there is no gap between partly filled degenerate orbitals).
    #[test]
    fn a_partly_filled_shell_shows_its_fractions_and_quotes_no_gap() {
        let third = 1.0 / 3.0;
        let rows = vec![
            row(Spin::Up, 0, 1, 0, -0.87, 1.0, idx(1)),
            row(Spin::Up, 1, 2, 1, -0.33, 1.0, idx(2)),
            row(Spin::Down, 0, 3, 0, -0.83, 1.0, idx(1)),
            row(Spin::Down, 1, 4, 1, -0.28, third, idx(2)),
            row(Spin::Down, 2, 5, 2, -0.28, third, idx(3)),
            row(Spin::Down, 3, 6, 3, -0.28, third, idx(4)),
            row(Spin::Down, 4, 7, 4, 0.05, 0.0, idx(5)),
        ];
        let levels = Levels::of(&rows);
        assert!(levels.fractional && levels.gap().is_none(), "{levels:?}");
        let line = orbitals::levels_line(&levels).expect("a HOMO line");
        assert!(line.starts_with("HOMO −0.280 Ha") && line.contains("partly filled shell: no HOMO–LUMO gap"), "{line}");
        assert!(!line.contains("Kohn–Sham gap"), "{line}");
        let v = RowView::of(&rows[3], &levels);
        assert_eq!(v.occupation, "0.33", "{v:?}");
        assert!(v.occupied);
        // an integer-occupation list is unchanged
        let h2 = Levels::of(&h2_rows());
        assert!(!h2.fractional && h2.gap().is_some());
    }

    /// ε = −0.5 Ha reads "−0.500" and "−13.61" (units::HARTREE_IN_EV); the tags the rows carry.
    #[test]
    fn rows_format_energies_in_ha_and_ev() {
        let rows = h2_rows();
        let levels = Levels::of(&rows);
        let v = RowView::of(&rows[0], &levels);
        assert_eq!((v.ha.as_str(), v.ev.as_str(), v.occupation.as_str()), ("−0.500", "−13.61", "1"), "{v:?}");
        assert_eq!(v.label, "#1");
        assert_eq!(orbitals::energy_ev(-0.5), format!("−{:.2}", 0.5 * units::HARTREE_IN_EV));
        // HOMO over both channels (the ↑ and ↓ σ_g are the same level: both tagged), LUMO = the σ_u
        let tags = |r: &OrbitalInfo| RowView::of(r, &levels).tags.iter().map(|t| t.text.clone()).collect::<Vec<_>>();
        assert_eq!(tags(&rows[0]), vec!["HOMO"]);
        assert_eq!(tags(&rows[2]), vec!["HOMO"]);
        assert_eq!(tags(&rows[1]), vec!["LUMO"]);
        let line = orbitals::levels_line(&levels).unwrap();
        assert!(line.contains("HOMO −0.500 Ha (−13.61 eV)") && line.contains("Kohn–Sham gap 0.520 Ha (14.15 eV)"), "{line}");
    }

    /// HOMO, LUMO and the gap come from the eigenvalues (`eps_range`: for a rotated set its lowest
    /// and highest), never from a rotated member's Rayleigh quotient.
    #[test]
    fn homo_lumo_and_the_gap_read_the_eigenvalues_of_a_rotated_set() {
        // an occupied pair (p_x, p_y) at −0.30 and −0.29, displayed rotated: the members' energies
        // are Rayleigh quotients (−0.296, −0.294) inside the span; an unoccupied pair above
        let mut a = row(Spin::Up, 0, 1, 0, -0.296, 1.0, idx(1));
        let mut b = row(Spin::Up, 1, 2, 1, -0.294, 1.0, idx(2));
        for r in [&mut a, &mut b] {
            r.eps_range = (-0.30, -0.29);
            r.rotated = true;
            r.cluster = Some((2, 0.01));
        }
        let mut u = row(Spin::Up, 2, 3, 2, 0.011, 0.0, idx(3));
        u.eps_range = (0.01, 0.03);
        u.rotated = true;
        u.residual = Some(2.5e-5);
        let rows = vec![a, b, u];
        let l = Levels::of(&rows);
        assert_eq!((l.homo, l.lumo), (Some(-0.29), Some(0.01)), "the span's ends, not the quotients");
        assert!((l.gap().unwrap() - 0.30).abs() < 1e-12);
        // both members of the occupied pair carry HOMO (their set contains the HOMO eigenvalue)
        for r in &rows[..2] {
            assert!(RowView::of(r, &l).tags.iter().any(|t| t.text == "HOMO"));
        }
        let tags = RowView::of(&rows[2], &l).tags;
        let rot = tags.iter().find(|t| t.text.starts_with("rotated for display")).expect("a rotated tag");
        assert!(rot.text.contains("eigensolve residual of the set 2.5e-5"), "{}", rot.text);
        assert!(rot.tip.contains("rotated within a degenerate set for display"));
        assert!(tags.iter().any(|t| t.text == "degenerate ×2 (spread 10.00 mHa)") || RowView::of(&rows[2], &l).tags.len() >= 2);
    }

    /// Clicking a row picks that orbital by its track — a row of the list, in the tab.
    #[test]
    fn picking_a_row_selects_by_track() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let snap = with_payload(&scene, Backend::Gpu, payload(h2_rows(), OrbitalRequest::List));
        let inp = input(&scene, Some(&snap), &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        st.show = ShowMode::Orbitals;
        st.open_tab(DockTab::Orbitals);
        let outs = click_widget(&ctx, &mut st, &inp, "orbital-↓-2");
        let sent = orbital_view_commands(&outs);
        assert_eq!(sent.len(), 1, "{sent:?}");
        assert_eq!(sent[0].what, OrbitalRequest::Orbital { spin: Spin::Down, track: 2, squared: false });
        assert_eq!(st.orbital_pick, Some((Spin::Down, 2)));
        // the tab's own show selector
        let outs = click_widget(&ctx, &mut st, &inp, "show-spin");
        assert_eq!(orbital_view_commands(&outs)[0].what, OrbitalRequest::SpinDensity);
    }

    /// Every pinned honesty string is in the text its widget shows (tooltips are built by these
    /// functions and constants; the layout test draws them).
    #[test]
    fn tooltips_carry_the_honesty_text() {
        let ks = "Kohn–Sham orbital: an auxiliary object of DFT, not a measured electron wavefunction. The total density is the physical observable.";
        let spin = "Spin is intrinsic (↑/↓), not a rotation. Orbitals are stationary: they move only with the nuclei.";
        assert!(orbitals::KS_NOTE.contains(ks) && orbitals::SPIN_NOTE.contains(spin));
        for m in [ShowMode::Orbitals, ShowMode::Coloured, ShowMode::Spin] {
            assert!(orbitals::mode_tip(m).contains(ks), "{m:?}: every orbital mode says what an orbital is");
        }
        assert!(orbitals::mode_tip(ShowMode::Spin).contains(spin));
        assert!(orbitals::mode_tip(ShowMode::Orbitals).contains("the sign of an orbital is arbitrary; only the relative sign of its lobes means anything"));
        assert!(orbitals::SIGN_TIP.contains("the sign of an orbital is arbitrary; only the relative sign of its lobes means anything"));
        assert!(orbitals::ROTATED_TIP.contains("rotated within a degenerate set for display"));
        assert!(orbitals::grid_split_tag(5e-4).text.contains("split by the grid by 0.50 mHa"));
        assert!(orbitals::BOX_STATE_TIP.contains("a state of the simulation box, not of the molecule"));
        assert!(orbitals::GAP_TIP.contains("the Kohn–Sham gap is not the fundamental gap"));
        assert!(orbitals::UNOCCUPIED_TIP.contains("The Kohn–Sham gap is not the fundamental gap"));
        assert_eq!(orbitals::CORE_TAG, "core: reconstructed for display");
        // the rows' tags use them
        let l = Levels::default();
        let mut r = row(Spin::Up, 0, 1, 0, -0.84, 1.0, OrbitalLabel::Atomic { n: 2, l: 1, m: "x" });
        r.rotated = true;
        r.grid_split = Some(7e-4);
        r.smeared_1s = true;
        let tags = RowView::of(&r, &l).tags;
        assert!(tags.iter().any(|t| t.tip.contains("rotated within a degenerate set for display")));
        assert!(tags.iter().any(|t| t.text.contains("split by the grid by 0.70 mHa")));
        assert!(tags.iter().any(|t| t.text.contains("smeared nucleus: wider than the physical 1s")));
        assert_eq!(RowView::of(&r, &l).label, "2p_x");
        let mut b = row(Spin::Up, 3, 9, 3, 0.04, 0.0, idx(4));
        b.box_state = true;
        let t = RowView::of(&b, &l).tags;
        assert!(t.iter().all(|t| !t.warn || t.tip.contains("a state of the simulation box, not of the molecule") || t.text.contains("not converged")));
        assert!(RowView::of(&b, &l).label_tip().contains(ks));
        // the lone-atom lines
        assert_eq!(orbitals::lone_atom_line(LoneAtomLabels::Labelled { shell_order_inverted: true }).as_deref(), Some("shell order inverted by the smeared nucleus — not physical (needs a finer grid or pseudopotentials)"));
        assert!(orbitals::lone_atom_line(LoneAtomLabels::Unlabelled("an open shell")).unwrap().contains("an open shell"));
        assert_eq!(orbitals::lone_atom_line(LoneAtomLabels::Labelled { shell_order_inverted: false }), None);
        assert_eq!(orbitals::lone_atom_line(LoneAtomLabels::NotLoneAtom), None);
    }

    /// The box-state tag is worded by the sign of ε (rule 7, Task 4 review M2).
    #[test]
    fn box_states_are_worded_by_the_sign_of_epsilon() {
        let mut up = row(Spin::Up, 2, 5, 2, 0.011, 0.0, idx(3));
        up.box_state = true;
        let mut bound = row(Spin::Up, 3, 6, 3, -0.0396, 0.0, idx(4));
        bound.box_state = true;
        let l = Levels::default();
        let t = |r: &OrbitalInfo| RowView::of(r, &l).tags.into_iter().find(|t| t.warn).expect("a box tag").text;
        assert_eq!(t(&up), "state of the box — unbound in LSDA at this box");
        assert_eq!(t(&bound), "bound, but its tail reaches the box walls: energy box-affected");
        // "not converged" with its residual
        let mut nc = row(Spin::Up, 4, 7, 4, 0.05, 0.0, idx(5));
        nc.not_converged = true;
        nc.residual = Some(3.2e-4);
        assert!(RowView::of(&nc, &l).tags.iter().any(|t| t.text == "not converged (residual 3.2e-4)"));
    }

    /// "show unoccupied": the request against what the payload brought, the computing state, the
    /// interrupted and held states.
    #[test]
    fn unoccupied_slider_shows_requested_and_shown() {
        let st = |requested: u8, shown: [usize; 2], hidden: [usize; 2]| UnoccupiedStatus { requested, shown, hidden, current: true, ..Default::default() };
        // m = 4 asked, a set cut by the block edge hid three, one is shown
        let lines = orbitals::unoccupied_lines(4, Some(&st(4, [1, 1], [3, 3])), false);
        assert!(lines.iter().any(|l| l.contains("4 requested, 1 shown")), "{lines:?}");
        assert!(lines.iter().any(|l| l.starts_with("↑: ")) && lines.iter().any(|l| l.starts_with("↓: ")), "{lines:?}");
        assert!(orbitals::unoccupied_lines(4, Some(&st(4, [4, 4], [0, 0])), false).is_empty(), "nothing hidden: nothing to say");
        // the provider's own block-edge note is not repeated
        let mut u = st(4, [1, 1], [3, 3]);
        u.notes = vec!["↑: 4 requested, 1 shown: a degenerate set cut by the block edge is hidden".into(), "unoccupied orbitals from an earlier geometry".into()];
        let lines = orbitals::unoccupied_lines(4, Some(&u), false);
        assert_eq!(lines.iter().filter(|l| l.contains("4 requested, 1 shown")).count(), 2, "one line per channel, from the counts: {lines:?}");
        assert!(lines.iter().any(|l| l == "unoccupied orbitals from an earlier geometry"));
        // a request the payload does not answer yet: computing (the eigensolve takes seconds)
        let lines = orbitals::unoccupied_lines(4, None, false);
        assert!(lines[0].starts_with("computing unoccupied orbitals…"), "{lines:?}");
        let lines = orbitals::unoccupied_lines(6, Some(&st(4, [4, 4], [0, 0])), false);
        assert!(lines[0].starts_with("computing unoccupied orbitals…"), "{lines:?}");
        // interrupted by a newer command; the driver retries on its own
        let mut u = st(4, [4, 4], [0, 0]);
        u.interrupted = true;
        assert!(orbitals::unoccupied_lines(4, Some(&u), false).contains(&"interrupted — recomputed when idle".to_string()));
        // held ≠ requested: the previous set is still shown
        let mut u = st(5, [4, 4], [0, 0]);
        u.held = 4;
        assert!(orbitals::unoccupied_lines(5, Some(&u), false).contains(&"showing the m = 4 set; 5 requested".to_string()));
        assert!(orbitals::unoccupied_lines(0, None, false).is_empty());
    }

    /// The spin readout, and the noise floor's wording.
    #[test]
    fn the_spin_readout_says_when_there_is_no_net_spin() {
        let li = SpinSummary { n_up: 2, n_dn: 1, integral: 1.0, abs_integral: 1.000528, max_abs: 0.4, rho_max: 13.0, scale: 0.4, noise: 1e-4 };
        let lines = orbitals::spin_lines(&li);
        assert_eq!(lines[0], "n↑ = 2, n↓ = 1");
        assert!(lines[1].contains("∫m = 1.0000"), "{lines:?}");
        assert!(lines[2].contains("∫|m| = 1.0005"), "{lines:?}");
        assert!(lines.last().unwrap().starts_with("max|m| / max ρ = 0.031"), "{lines:?}");
        let h2 = SpinSummary { n_up: 1, n_dn: 1, integral: 0.0, abs_integral: 2e-6, max_abs: 2.4e-7 * 0.4, rho_max: 0.4, scale: 1e-3 * 0.4, noise: 0.0 };
        let lines = orbitals::spin_lines(&h2);
        assert!(lines.iter().any(|l| l.starts_with("no net spin (max|m| = 2.4e-7 · max ρ)")), "no budget known: the floored scale's wording; {lines:?}");
        // with the solver's noise budget: the closed shell is not drawn, and the readout says so
        let closed = SpinSummary { noise: 1.4e-4, ..h2 };
        let lines = orbitals::spin_lines(&closed);
        assert!(lines.iter().any(|l| l.starts_with("m is within the solver's noise (a closed shell has m = 0): ∫|m| = 2.0e-6 e is inside the noise budget 1.4e-4 e")), "{lines:?}");
        assert!(!lines.iter().any(|l| l.starts_with("max|m| / max ρ")), "{lines:?}");
        // a real spin never says it: the lone Li above (noise budget 1e-4 against ∫|m| = 1)
        assert!(!orbitals::spin_lines(&li).iter().any(|l| l.contains("no net spin")));
        let broken = SpinSummary { n_up: 1, n_dn: 1, integral: 0.0, abs_integral: 0.6, max_abs: 0.1, rho_max: 0.2, scale: 0.1, noise: 1.4e-4 };
        let lines = orbitals::spin_lines(&broken);
        assert!(lines.iter().any(|l| l.contains("spin of both signs")) && !lines.iter().any(|l| l.contains("no net spin")), "{lines:?}");
    }

    /// Estimated mode: the orbitals are the last background DFT solve's, faded and labelled; the
    /// current step's own are not. The text over the view for each state in which there is nothing
    /// to draw.
    #[test]
    fn estimated_mode_fades_and_labels_orbitals() {
        let scene = Scene::preset("h2").unwrap();
        let mut st = UiState::new("h2", &scene);
        st.show = ShowMode::Orbitals;
        let mut snap = with_payload(&scene, Backend::Estimated, payload(h2_rows(), OrbitalRequest::List));
        let mut m = model();
        m.audits_background = true;
        snap.model = Some(m.clone());
        snap.dynamics.orbitals_t = units::fs_to_au_time(12.5);
        snap.dynamics.orbitals_side = Some(SideKind::Refresh);
        snap.dynamics.step = 3;
        let mut shown = OrbitalShown::default();
        shown.follow(&snap.dynamics);
        assert_eq!(shown.fade(), SURROGATE_DENSITY_FADE);
        let label = shown.label(&st, Some(&snap)).expect("a label");
        assert!(label.starts_with("orbitals from t = 12.50 fs (last DFT solve)"), "{label}");
        assert!(label.contains("The estimated solver has no electrons"), "{label}");
        // a DFT step's own payload: full strength, no label (current geometry, playing or paused)
        snap.backend = Backend::Gpu;
        snap.dynamics.orbitals_side = None;
        snap.dynamics.orbitals.as_mut().unwrap().positions = scene.nuclei.iter().map(|n| n.pos).collect();
        shown.follow(&snap.dynamics);
        assert_eq!(shown.fade(), 1.0);
        assert_eq!(shown.label(&st, Some(&snap)), None);
        // paused on a geometry the payload was not solved at
        snap.dynamics.paused = true;
        snap.dynamics.orbitals.as_mut().unwrap().positions = vec![[9.0, 0.0, 0.0]; scene.nuclei.len()];
        assert!(shown.label(&st, Some(&snap)).unwrap().contains("(an earlier geometry)"));
        // nothing to draw: no payload yet, no electrons, the CPU teacher
        let none = snapshot(&scene, Backend::Gpu, None);
        assert_eq!(shown.label(&st, Some(&none)).as_deref(), Some(ORBITALS_NEED_DFT));
        assert_eq!(shown.label(&st, None).as_deref(), Some(ORBITALS_NEED_DFT));
        let empty = Scene::preset("empty").unwrap();
        assert_eq!(shown.label(&st, Some(&snapshot(&empty, Backend::Gpu, None))).as_deref(), Some(ORBITALS_NO_ELECTRONS));
        let mut cpu_teacher = snapshot(&scene, Backend::Estimated, Some(model()));
        cpu_teacher.model.as_mut().unwrap().audits_background = false;
        let say = shown.label(&st, Some(&cpu_teacher)).unwrap();
        assert_eq!(say, ORBITALS_CPU_TEACHER);
        assert!(say.contains("no background solver") && say.contains("switch the solver to GPU"), "{say}");
        // and none at all in Density mode
        st.show = ShowMode::Density;
        assert_eq!(shown.label(&st, Some(&cpu_teacher)), None);
    }

    /// Ctrl K finds the modes by their names.
    #[test]
    fn search_finds_the_modes() {
        assert_eq!(search::results("spin", 3)[0].name, "Spin density");
        assert_eq!(search::results("orbitals", 1)[0].name, "Orbitals");
        assert_eq!(search::results("all orbitals", 1)[0].name, "All orbitals coloured");
        assert_eq!(search::results("density", 1)[0].name, "Density");
        for name in ["Density", "Orbitals", "All orbitals coloured", "Spin density"] {
            assert!(search::ENTRIES.iter().any(|e| e.name == name && matches!(e.place, search::Place::Menu(menus::MenuId::View))), "{name}");
        }
        assert_eq!(search::results("unoccupied", 1)[0].place, search::Place::Dock(DockTab::Orbitals));
    }

    /// The legend names the colour pair or the palette in use.
    #[test]
    fn the_legend_names_the_colours_in_use() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let mut st = UiState::new("h2", &scene);
        let field = |signed| Some(sim::OrbitalField::Cpu { values: std::sync::Arc::new(vec![0.0; 27]), signed, scale: 1.0 });
        let mut o = payload(h2_rows(), OrbitalRequest::Orbital { spin: Spin::Up, track: 1, squared: false });
        o.field = field(true);
        let snap = with_payload(&scene, Backend::Gpu, o.clone());
        let inp = input(&scene, Some(&snap), &cam);
        assert!(legend_of(&st, &inp).is_none(), "Density mode: no legend");
        st.show = ShowMode::Orbitals;
        let l = legend_of(&st, &inp).unwrap();
        assert_eq!(l.swatches.iter().map(|s| s.1.as_str()).collect::<Vec<_>>(), ["+", "−"]);
        assert!(l.text.contains("only the relative sign of its lobes means anything") && l.tip.contains(orbitals::KS_NOTE));
        // ψ̃²: no swatches
        o.field = field(false);
        let snap = with_payload(&scene, Backend::Gpu, o);
        let inp = input(&scene, Some(&snap), &cam);
        assert!(legend_of(&st, &inp).unwrap().swatches.is_empty());
        // spin
        st.show = ShowMode::Spin;
        let mut o = payload(h2_rows(), OrbitalRequest::SpinDensity);
        o.field = field(true);
        let snap = with_payload(&scene, Backend::Gpu, o);
        let inp = input(&scene, Some(&snap), &cam);
        let l = legend_of(&st, &inp).unwrap();
        assert_eq!(l.swatches.iter().map(|s| s.1.as_str()).collect::<Vec<_>>(), ["↑ excess", "↓ excess"]);
        assert!(l.tip.contains("Spin is intrinsic (↑/↓), not a rotation."));
        // coloured: one swatch per colour slot (↓ shares ↑'s), the scale in the text
        st.show = ShowMode::Coloured;
        let mut o = payload(h2_rows(), OrbitalRequest::AllColoured { include: None, physical: false, look: Default::default() });
        o.field = Some(sim::OrbitalField::CpuColour { rgba: std::sync::Arc::new(vec![[0.0; 4]; 27]), scale: 1.0 });
        let snap = with_payload(&scene, Backend::Gpu, o);
        let inp = input(&scene, Some(&snap), &cam);
        let l = legend_of(&st, &inp).unwrap();
        assert_eq!(l.swatches.len(), 1, "σ_g↑ and σ_g↓ share slot 0; σ_u is unoccupied and not included");
        assert_eq!(l.text, "each orbital scaled to its own peak");
    }

    /// The chrome lays out with a payload in every mode and tab, sends nothing without input, and
    /// shows the tab for each state (the unoccupied lines, the spin readout, the notes).
    #[test]
    fn the_orbitals_tab_draws_in_every_mode_and_sends_nothing() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        st.open_tab(DockTab::Orbitals);
        let mut u = UnoccupiedStatus { requested: 4, shown: [1, 1], hidden: [3, 3], current: true, interrupted: true, held: 3, ..Default::default() };
        u.notes = vec!["a note".into()];
        let mut o = payload(h2_rows(), OrbitalRequest::Orbital { spin: Spin::Up, track: 1, squared: false });
        o.unoccupied = Some(u);
        o.notes = vec!["selected orbital no longer exists — showing #1".into()];
        o.lone_atom = LoneAtomLabels::Labelled { shell_order_inverted: true };
        o.spin = SpinSummary { n_up: 1, n_dn: 1, ..Default::default() };
        o.rows[1].box_state = true;
        o.rows[1].not_converged = true;
        o.rows[1].residual = Some(1e-4);
        o.rows[0].grid_split = Some(4e-4);
        o.rows[0].smeared_1s = true;
        let snaps = [None, Some(with_payload(&scene, Backend::Gpu, o)), Some(with_payload(&scene, Backend::Gpu, payload(Vec::new(), OrbitalRequest::List)))];
        for snap in &snaps {
            for mode in ShowMode::ALL {
                st.show = mode;
                for (w, label) in [(1400.0, None), (1000.0, Some("orbitals from t = 1.00 fs (last DFT solve). The estimated solver has no electrons".to_string()))] {
                    let inp = PanelInput { orbital_label: label, ..input(&scene, snap.as_ref(), &cam) };
                    for _ in 0..2 {
                        let out = frame(&ctx, &mut st, &inp, w, vec![]);
                        assert!(out.commands.is_empty() && out.error.is_none(), "{mode:?} at {w}");
                    }
                }
            }
        }
    }

    // ---- review round 1 ----

    /// Every text of the frame with its y: the galleys of the text shapes, nested shapes included.
    fn frame_texts(ctx: &egui::Context, st: &mut UiState, inp: &PanelInput<'_>, events: Vec<egui::Event>, time: f64) -> Vec<(String, f32)> {
        fn walk(s: &egui::epaint::Shape, out: &mut Vec<(String, f32)>) {
            match s {
                egui::epaint::Shape::Text(t) => out.push((t.galley.text().to_string(), t.pos.y)),
                egui::epaint::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                _ => {}
            }
        }
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0))),
            events,
            time: Some(time),
            ..Default::default()
        };
        let mut full = ctx.run_ui(raw, |root| {
            show(root, st, inp);
        });
        full.textures_delta.clear();
        let mut out = Vec::new();
        full.shapes.iter().for_each(|c| walk(&c.shape, &mut out));
        out
    }

    /// Everything that is on screen once the pointer has rested on the probed widget long enough
    /// for its tooltip: the widget's own hover text, as the user reads it.
    fn hover_text(ctx: &egui::Context, st: &mut UiState, inp: &PanelInput<'_>, probe: &str) -> String {
        for _ in 0..3 {
            frame(ctx, st, inp, 1400.0, vec![]);
        }
        let c = probed_rect(probe).center();
        let mut all = String::new();
        for k in 0..8 {
            let events = if k == 0 { vec![egui::Event::PointerMoved(c)] } else { vec![] };
            for (s, _) in frame_texts(ctx, st, inp, events, 10.0 + 0.3 * k as f64) {
                all += &s;
                all += "\n";
            }
        }
        all
    }

    /// The pinned honesty strings are in the text the widgets show on hover (not just in the
    /// constants): the mode buttons, a row's label.
    #[test]
    fn the_widgets_show_the_honesty_text_on_hover() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let snap = with_payload(&scene, Backend::Gpu, payload(h2_rows(), OrbitalRequest::List));
        let inp = input(&scene, Some(&snap), &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        st.show = ShowMode::Orbitals;
        st.open_tab(DockTab::Orbitals);
        let t = hover_text(&ctx, &mut st, &inp, "show-orbitals");
        assert!(t.contains("the sign of an orbital is arbitrary; only the relative sign of its lobes means anything"), "{t}");
        assert!(t.contains("an auxiliary object of DFT"), "{t}");
        let t = hover_text(&ctx, &mut st, &inp, "show-spin");
        assert!(t.contains("Spin is intrinsic (↑/↓), not a rotation"), "{t}");
        let t = hover_text(&ctx, &mut st, &inp, "orbital-↑-1");
        assert!(t.contains("an auxiliary object of DFT"), "{t}");
    }

    /// The Ne shell inversion is the first thing under the levels line — above every orbital row,
    /// so it is on screen with the list — and the "HOMO" of such an atom says it is not the physical
    /// one, in its tag, its tooltip and the levels line. Molecules with a heavy nucleus get the
    /// smeared-core note in the same place.
    #[test]
    fn the_inversion_warning_sits_above_the_rows_and_qualifies_the_homo() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        st.show = ShowMode::Orbitals;
        st.open_tab(DockTab::Orbitals);
        let mut rows = vec![
            row(Spin::Up, 0, 1, 0, -9.66, 1.0, OrbitalLabel::Atomic { n: 1, l: 0, m: "" }),
            row(Spin::Up, 1, 2, 1, -0.82, 1.0, OrbitalLabel::Atomic { n: 2, l: 1, m: "x" }),
            row(Spin::Up, 2, 5, 4, -0.39, 1.0, OrbitalLabel::Atomic { n: 2, l: 0, m: "" }),
        ];
        rows[0].smeared_1s = true;
        let mut o = payload(rows, OrbitalRequest::Orbital { spin: Spin::Up, track: 2, squared: false });
        o.lone_atom = LoneAtomLabels::Labelled { shell_order_inverted: true };
        let levels = Levels::of(&o.rows).with_lone_atom(o.lone_atom);
        let line = orbitals::levels_line(&levels).unwrap();
        assert!(line.starts_with("HOMO −0.390 Ha") && line.ends_with("shell order inverted: not the physical HOMO"), "{line}");
        let homo = RowView::of(&o.rows[2], &levels).tags.into_iter().find(|t| t.text.starts_with("HOMO")).expect("the 2s is the HOMO");
        assert!(homo.text.contains("not the physical HOMO") && homo.warn && homo.tip.contains(orbitals::HOMO_INVERTED_TIP), "{homo:?}");
        assert!(!orbitals::HOMO_TIP.contains("only one with a physical meaning"), "the HOMO tooltip carries the rule-7 caveat");
        assert!(orbitals::HOMO_TIP.contains("far too shallow"));
        // on screen: the warning's y is above the first row's
        let snap = with_payload(&scene, Backend::Gpu, o);
        let inp = input(&scene, Some(&snap), &cam);
        for _ in 0..3 {
            frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        }
        let texts = frame_texts(&ctx, &mut st, &inp, vec![], 5.0);
        let y_of = |needle: &str| texts.iter().find(|(t, _)| t.contains(needle)).map(|(_, y)| *y).unwrap_or_else(|| panic!("no text {needle:?} on screen"));
        assert!(y_of("shell order inverted by the smeared nucleus — not physical") < y_of("1s ↑"), "the warning is drawn above the rows");
        assert!(y_of("HOMO −0.390 Ha") < y_of("1s ↑"));
        // a molecule with a heavy nucleus: the note is there; none for a lone atom or a light one
        let l = orbitals::smeared_molecule_line(LoneAtomLabels::NotLoneAtom, 1.2).expect("Zσ = 1.2 ≥ 0.5");
        assert!(l.contains("smeared nuclei") && l.contains("core"), "{l}");
        assert_eq!(orbitals::smeared_molecule_line(LoneAtomLabels::NotLoneAtom, 0.3), None);
        assert_eq!(orbitals::smeared_molecule_line(LoneAtomLabels::Labelled { shell_order_inverted: false }, 1.5), None);
    }

    /// A split below the display's resolution is not announced as a split of 0.00 mHa; the legend
    /// speaks of an iso level only when the isosurface is drawn.
    #[test]
    fn small_splits_and_the_legends_iso_clause() {
        assert_eq!(orbitals::grid_split_tag(1.6e-8).text, "not split by the grid (< 0.005 mHa)");
        assert_eq!(orbitals::grid_split_tag(4.97e-4).text, "split by the grid by 0.50 mHa");
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let mut st = UiState::new("h2", &scene);
        st.show = ShowMode::Orbitals;
        let mut o = payload(h2_rows(), OrbitalRequest::Orbital { spin: Spin::Up, track: 1, squared: false });
        o.field = Some(sim::OrbitalField::Cpu { values: std::sync::Arc::new(vec![0.0; 27]), signed: true, scale: 1.0 });
        let snap = with_payload(&scene, Backend::Gpu, o);
        let inp = input(&scene, Some(&snap), &cam);
        st.volume_mode = crate::ui::VOLUME_VOLUME;
        assert!(!legend_of(&st, &inp).unwrap().text.contains("iso"), "no isosurface is drawn in Volume mode");
        st.volume_mode = crate::ui::VOLUME_ISO;
        assert!(legend_of(&st, &inp).unwrap().text.contains("iso ±0.15"));
    }

    /// The provider's "selected orbital no longer exists" note is shown for a pick the user made
    /// and dropped for one the viewer adopted from the HOMO placeholder.
    #[test]
    fn the_vanished_selection_note_is_not_shown_for_an_adopted_pick() {
        let mut o = payload(h2_rows(), OrbitalRequest::List);
        o.notes = vec!["selected orbital no longer exists — showing #1 ↑".into(), "another note".into()];
        assert_eq!(orbitals::visible_notes(&o, false).len(), 2);
        assert_eq!(orbitals::visible_notes(&o, true), vec![&"another note".to_string()]);
        // picking a row makes the pick the user's again
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let snap = with_payload(&scene, Backend::Gpu, payload(h2_rows(), OrbitalRequest::List));
        let inp = input(&scene, Some(&snap), &cam);
        let mut st = UiState::new("h2", &scene);
        st.orbital_pick_adopted = true;
        act(&mut st, &inp, Action::PickOrbital { spin: Spin::Up, track: 1 });
        assert!(!st.orbital_pick_adopted);
    }

    /// Final review I3: the dots banner is worded by what the dots sample.
    #[test]
    fn the_dots_banner_says_what_the_dots_sample() {
        let scene = Scene::preset("h2").unwrap();
        let mut st = UiState::new("h2", &scene);
        st.show = ShowMode::Density;
        assert!(orbitals::dots_note(&st).starts_with("dots are samples of where an electron is likely to be"));
        st.show = ShowMode::Orbitals;
        assert!(orbitals::dots_note(&st).starts_with("dots are samples of where an electron is likely to be"));
        st.show = ShowMode::Coloured;
        let peak = orbitals::dots_note(&st);
        assert!(peak.contains("own peak") && peak.contains("not the electron density"), "{peak}");
        st.colour_physical = true;
        let phys = orbitals::dots_note(&st);
        assert!(phys.contains("density summed over the occupied orbitals"), "{phys}");
        st.included = Some(Default::default());
        assert!(orbitals::dots_note(&st).contains("included orbitals"));
        st.show = ShowMode::Spin;
        let spin = orbitals::dots_note(&st);
        assert!(spin.contains("|ρ↑ − ρ↓|") && spin.contains("not electrons") && !spin.contains("likely to be"), "{spin}");
        // and it is the banner over the view in dots mode
        let cam = camera();
        st.volume_mode = crate::ui::VOLUME_DOTS;
        let notes = overlays::chip_notes(&st, &input(&scene, None, &cam));
        assert!(notes.iter().any(|(_, n)| n == &spin), "{notes:?}");
    }

    /// Final review I4: a gap whose LUMO is a state of the box, unconverged or an earlier
    /// geometry's is not offered as the molecule's.
    #[test]
    fn the_levels_line_qualifies_a_gap_that_is_the_boxs() {
        let mut rows = h2_rows(); // σ_g ↑ −0.5, σ_u ↑ +0.02 (LUMO), σ_g ↓ −0.5
        let clean = orbitals::levels_line(&Levels::of(&rows)).unwrap();
        assert!(clean.contains("Kohn–Sham gap") && !clean.contains("box size"), "{clean}");
        rows[1].box_state = true;
        let line = orbitals::levels_line(&Levels::of(&rows)).unwrap();
        assert!(line.contains("the LUMO is a state of the box, so this gap depends on the box size"), "{line}");
        rows[1].not_converged = true;
        assert!(orbitals::levels_line(&Levels::of(&rows)).unwrap().contains("the LUMO is not converged"));
        let stale = UnoccupiedStatus { requested: 4, current: false, ..Default::default() };
        let line = orbitals::levels_line(&Levels::of(&h2_rows()).with_unoccupied(Some(&stale))).unwrap();
        assert!(line.contains("earlier geometry"), "{line}");
        let current = UnoccupiedStatus { requested: 4, current: true, ..Default::default() };
        assert!(!orbitals::levels_line(&Levels::of(&h2_rows()).with_unoccupied(Some(&current))).unwrap().contains("earlier"));
        // computing: the same-m solve is announced while a request is out
        let u = UnoccupiedStatus { requested: 4, shown: [4, 4], current: true, ..Default::default() };
        assert!(orbitals::unoccupied_lines(4, Some(&u), true)[0].starts_with("computing unoccupied orbitals…"));
        assert!(orbitals::unoccupied_lines(4, Some(&u), false).is_empty());
    }
}

/// The Activity tab (activity spec §3.3): state, job rows, the solve log, the views contract.
mod activity_tab {
    use super::*;
    use sim::activity::{TeachOrigin, TeachOutcome, TeachRule};
    use std::sync::Arc;

    pub(super) fn event(origin: TeachOrigin, rule: TeachRule, outcome: TeachOutcome, step: Option<u64>) -> sim::TeachEvent {
        let mut e = sim::TeachEvent::new(origin, rule, outcome);
        e.step = step;
        e.n_atoms = 2;
        e.solve_ms = 92.0;
        e.gap = Some(3.1e-3);
        e
    }

    /// A few events of every origin, rule and outcome the log words (for the smoke runs).
    pub(super) fn every_kind_of_event() -> Vec<sim::TeachEvent> {
        use sim::activity::{PassJob, TrustKey};
        use sim::activity::{TeachOrigin as O, TeachOutcome as K, TeachRule as R};
        let mut v = vec![
            event(O::Pass, R::Pass(PassJob::Reference { z: 1, multiplicity: 2 }), K::Reference, None),
            event(O::Pass, R::Pass(PassJob::Scan), K::Kept, None),
            event(O::Pass, R::Pass(PassJob::HotBurst), K::KeptBoxAffected, None),
            event(O::Pass, R::Cell, K::NotSolved, None),
            event(O::Fallback, R::Untrusted(TrustKey::EnergyResidual), K::Kept, Some(3)),
            event(O::Fallback, R::NewEnvironment, K::Thinned, Some(4)),
            event(O::Fallback, R::Spread, K::BoxNotKept, Some(5)),
            event(O::Fallback, R::NonFiniteAnswer, K::Failed, Some(6)),
            event(O::Audit, R::Audit, K::KeptByVote, Some(7)),
            event(O::Verification, R::RelaxVerification, K::Kept, Some(8)),
        ];
        v[2].burst_t = Some(1830.0);
        v[2].box_affected = true;
        v[2].atom = Some(0);
        v[4].detail = "the model was not trusted: max training energy residual 2.2e-3 > 1.0e-3 Ha/atom".into();
        v[8].audit = Some(sim::AuditVerdict::Passed);
        v[8].audit_lag = Some(40);
        v
    }

    /// The log is newest first, counts solves (not skips), keeps 5 000 rows and keeps counting
    /// (Review Focus 2).
    #[test]
    fn the_log_keeps_5000_rows_and_the_counters_keep_counting() {
        use sim::activity::{TeachOrigin as O, TeachOutcome as K, TeachRule as R};
        let scene = Scene::preset("h2").unwrap();
        let mut a = activity::ActivityState::new();
        let mut snap = snapshot(&scene, Backend::Estimated, Some(model()));
        snap.activity.events = (0..6000).map(|i| event(O::Fallback, R::Grade, K::Kept, Some(i))).collect();
        snap.activity.events.push(event(O::Pass, R::Cell, K::NotSolved, None));
        a.absorb(&snap);
        assert_eq!(a.rows.len(), activity::LOG_ROWS);
        assert_eq!(a.session_solves, 6000, "a skip is not a solve");
        assert_eq!(a.rows.back().unwrap().event.rule, R::Cell, "newest at the back, drawn first");
    }

    /// Plan review S6: the burst-temperature histogram covers the whole session, apart from the
    /// 5 000-row ring that drops early rows.
    #[test]
    fn burst_temperatures_are_kept_for_the_whole_session() {
        use sim::activity::{PassJob, TeachOrigin as O, TeachOutcome as K, TeachRule as R};
        let scene = Scene::preset("h2").unwrap();
        let mut a = activity::ActivityState::new();
        let mut snap = snapshot(&scene, Backend::Estimated, Some(model()));
        snap.activity.events = (0..10)
            .map(|i| {
                let mut e = event(O::Pass, R::Pass(PassJob::HotBurst), K::Kept, None);
                e.burst_t = Some(1850.0 + i as f64);
                e
            })
            .collect();
        a.absorb(&snap);
        snap.activity.events = (0..6000).map(|i| event(O::Fallback, R::Grade, K::Kept, Some(i))).collect();
        a.absorb(&snap);
        assert!(a.rows.iter().all(|r| r.event.burst_t.is_none()), "the ring dropped the bursts");
        assert_eq!(a.burst_hist.iter().sum::<u32>(), 10);
        assert_eq!(a.burst_hist[(1850.0 / activity::BURST_BIN_K) as usize], 10);
    }

    /// Plan review M6: the step cell is the driver step of the event's evaluation; an audit says
    /// how many answered steps after its dispatch it was judged, and no dispatch step is invented.
    #[test]
    fn the_step_cell_says_what_it_means() {
        use activity::log::step_cell;
        use sim::activity::{PassJob, TeachOrigin as O, TeachOutcome as K, TeachRule as R};
        assert_eq!(step_cell(&event(O::Pass, R::Pass(PassJob::Scan), K::Kept, None)), ("pass".to_string(), None));
        assert_eq!(step_cell(&event(O::Fallback, R::Grade, K::Kept, Some(1607))), ("1607".to_string(), None));
        let mut a = event(O::Audit, R::Audit, K::Kept, Some(1640));
        a.audit_lag = Some(40);
        assert_eq!(step_cell(&a), ("1640".to_string(), Some("judged 40 answered steps after its dispatch".to_string())));
        // an event of a stepping origin that was never stamped says so, rather than "pass"
        assert_eq!(step_cell(&event(O::Fallback, R::Grade, K::Kept, None)).0, "—");
    }

    /// Solves this session add up over workers; a respawn does not reset them.
    #[test]
    fn session_solves_survive_a_respawn() {
        use sim::activity::{TeachOrigin as O, TeachOutcome as K, TeachRule as R};
        let scene = Scene::preset("h2").unwrap();
        let mut a = activity::ActivityState::new();
        let mut s1 = snapshot(&scene, Backend::Estimated, Some(model()));
        s1.activity.events = vec![event(O::Fallback, R::NoModel, K::Kept, Some(1))];
        a.absorb(&s1);
        let mut s2 = snapshot(&scene, Backend::Gpu, None);
        s2.activity.events = Vec::new();
        a.absorb(&s2);
        let mut s3 = snapshot(&scene, Backend::Estimated, Some(model()));
        s3.activity.events = vec![event(O::Audit, R::Audit, K::Kept, Some(9))];
        a.absorb(&s3);
        assert_eq!(a.session_solves, 2);
    }

    /// The job rows and their controls (spec §3.3.1): a pass row cancels, a playing solve pauses,
    /// an audit has no control and says why.
    #[test]
    fn job_rows_carry_the_existing_controls() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let mut snap = snapshot(&scene, Backend::Estimated, Some(model()));
        snap.dynamics.sampling = Some(sim::SamplingProgress { done: 41, total: 273, stage: "scans", last_energy: -1.1, cancelled: false, skipped: 0, failed: 0, finished: false, target: None });
        let probe = sim::ProbeView { side: sim::activity::SlotView { running: true, scf_step: 4, elapsed_s: 0.2, solves: 3, ..Default::default() }, ..Default::default() };
        let inp = PanelInput { probe: Some(probe), ..input(&scene, Some(&snap), &cam) };
        let rows = activity::jobs::rows(&inp, inp.probe.as_ref());
        let pass = rows.iter().find(|r| r.key == activity::jobs::JobKey::Pass).unwrap();
        assert!(matches!(pass.progress, activity::jobs::JobProgress::Fraction { done: 41, total: 273 }));
        assert!(matches!(pass.control, Some((_, Action::CancelSampling))));
        let audit = rows.iter().find(|r| r.key == activity::jobs::JobKey::Audit).unwrap();
        assert!(audit.control.is_none() && audit.no_control_why.is_some());
        assert!(audit.detail.contains("iteration 4"));
    }

    /// A playing run's solve pauses; a paused one only says why not; a refit, a relax and the
    /// first-pass offer carry their own controls.
    #[test]
    fn the_other_jobs_have_their_own_rows() {
        use activity::jobs::{rows, JobKey};
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let mut snap = snapshot(&scene, Backend::Estimated, Some(ModelStatus { audit_in_flight: false, ..model() }));
        let live = |step| sim::ProbeView { main: sim::activity::SlotView { running: true, scf_step: step, elapsed_s: 0.4, solves: 1, ..Default::default() }, ..Default::default() };
        let playing = PanelInput { playing: true, probe: Some(live(7)), ..input(&scene, Some(&snap), &cam) };
        let r = rows(&playing, playing.probe.as_ref());
        let solve = r.iter().find(|r| r.key == JobKey::LiveSolve).unwrap();
        assert_eq!(solve.title, "DFT fallback");
        assert!(matches!(solve.control, Some((_, Action::Pause))));
        let paused = PanelInput { playing: false, probe: Some(live(7)), ..input(&scene, Some(&snap), &cam) };
        let r = rows(&paused, paused.probe.as_ref());
        let solve = r.iter().find(|r| r.key == JobKey::LiveSolve).unwrap();
        assert!(solve.control.is_none() && solve.no_control_why.is_some());
        let refit = PanelInput { probe: Some(sim::ProbeView { refit: Some((0.2, 250)), ..Default::default() }), ..input(&scene, Some(&snap), &cam) };
        let r = rows(&refit, refit.probe.as_ref());
        assert!(r.iter().any(|r| r.key == JobKey::Refit && r.detail == "250 samples"));
        // the first-pass offer waits for "Start anyway"
        snap.model.as_mut().unwrap().pass_offer = Some(PassOffer { plan: sim::SamplingPlan::default(), solves: 365, eta_s: 366.0, per_solve_s: 1.0, full_solves: 512, full_eta_s: 512.0 });
        let offered = input(&scene, Some(&snap), &cam);
        let r = rows(&offered, None);
        let o = r.iter().find(|r| r.key == JobKey::PassOffer).unwrap();
        assert!(matches!(&o.control, Some((l, Action::StartOffered(_))) if l == "Start anyway"));
        // nothing runs: no rows
        let idle = snapshot(&scene, Backend::Gpu, None);
        assert!(rows(&input(&scene, Some(&idle), &cam), Some(&sim::ProbeView::default())).is_empty());
    }

    /// Task 5 review I2: the views are dropped when a snapshot has no fitted model (a Forget) and
    /// kept across snapshots whose `views` is `None` while the model is fitted.
    #[test]
    fn views_are_dropped_without_a_fitted_model() {
        let scene = Scene::preset("h2").unwrap();
        let mut st = activity::ActivityState::new();
        let mut s = snapshot(&scene, Backend::Estimated, Some(model()));
        s.activity.views = Some(Arc::new(activity::fixture_views()));
        st.absorb(&s);
        assert!(st.views.is_some());
        s.activity.views = None;
        st.absorb(&s);
        assert!(st.views.is_some(), "None = no new build");
        st.absorb(&snapshot(&scene, Backend::Estimated, Some(ModelStatus { fitted: false, ..model() })));
        assert!(st.views.is_none(), "forgotten");
        s.activity.views = Some(Arc::new(activity::fixture_views()));
        st.absorb(&s);
        st.absorb(&snapshot(&scene, Backend::Gpu, None));
        assert!(st.views.is_none(), "no model at all");
    }

    /// Task 5 review M2: the run divider goes by event time. An old-run event carried by the
    /// snapshot that announces the new run is placed before the divider, the new run's after it.
    #[test]
    fn the_run_divider_follows_event_time() {
        use activity::log::{items, Item};
        let scene = Scene::preset("h2").unwrap();
        let mut st = activity::ActivityState::new();
        let first = sim::TeachEvent { step: Some(39), ..sim::TeachEvent::new(TeachOrigin::Fallback, TeachRule::Spread, TeachOutcome::Kept) };
        let old = sim::TeachEvent { step: Some(40), ..sim::TeachEvent::new(TeachOrigin::Fallback, TeachRule::Spread, TeachOutcome::Kept) };
        std::thread::sleep(Duration::from_millis(2));
        let run = Instant::now();
        std::thread::sleep(Duration::from_millis(2));
        let new = sim::TeachEvent { step: Some(0), ..sim::TeachEvent::new(TeachOrigin::Fallback, TeachRule::Spread, TeachOutcome::Kept) };
        let mut s = snapshot(&scene, Backend::Estimated, Some(model()));
        s.activity.events = vec![first, old, new];
        s.activity.run_started = Some(run);
        st.absorb(&s);
        let r = *st.runs.last().unwrap();
        assert!(st.rows[1].at < r && st.rows[2].at >= r);
        // the divider sits between the old run's last row and the new run's first, and none
        // before the session's first row
        assert_eq!(items(&st), vec![Item::Row(0), Item::Row(1), Item::Divider, Item::Row(2)]);
    }

    /// The chips filter by origin and by kept / not kept.
    #[test]
    fn the_chips_filter_the_log() {
        use activity::log::{admits, items, Item};
        use sim::activity::{TeachOrigin as O, TeachOutcome as K, TeachRule as R};
        let scene = Scene::preset("h2").unwrap();
        let mut st = activity::ActivityState::new();
        let mut s = snapshot(&scene, Backend::Estimated, Some(model()));
        s.activity.events = vec![
            event(O::Fallback, R::Grade, K::Kept, Some(1)),
            event(O::Fallback, R::Grade, K::Thinned, Some(2)),
            event(O::Audit, R::Audit, K::Kept, Some(3)),
            event(O::Pass, R::Pass(sim::activity::PassJob::Scan), K::Kept, None),
        ];
        st.absorb(&s);
        assert_eq!(items(&st).len(), 4);
        st.filter.audit = false;
        st.filter.not_kept = false;
        assert_eq!(items(&st), vec![Item::Row(0), Item::Row(3)]);
        assert!(admits(&st.filter, &st.rows[0].event) && !admits(&st.filter, &st.rows[1].event));
    }

    /// Review Focus 5: opening Training sends SetActivityViews(true) once; a respawn (views_sent
    /// reset by the viewer) sends it again; nothing is sent while the section stays closed.
    #[test]
    fn the_views_command_is_resent_after_a_worker_respawn() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let ctx = egui::Context::default();
        let mut st = UiState::new("h2", &scene);
        st.open_tab(DockTab::Activity);
        let inp = input(&scene, None, &cam);
        assert!(frame(&ctx, &mut st, &inp, 1400.0, vec![]).commands.is_empty());
        st.activity.training_open = true;
        let out = frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        assert!(matches!(out.commands.as_slice(), [Command::SetActivityViews(true)]));
        assert!(frame(&ctx, &mut st, &inp, 1400.0, vec![]).commands.is_empty());
        st.activity.views_sent = false; // Viewer::start
        assert!(matches!(frame(&ctx, &mut st, &inp, 1400.0, vec![]).commands.as_slice(), [Command::SetActivityViews(true)]));
        // closing the dock turns them off again
        st.left_open = false;
        assert!(matches!(frame(&ctx, &mut st, &inp, 1400.0, vec![]).commands.as_slice(), [Command::SetActivityViews(false)]));
    }
}

/// The plain wording of rules and outcomes (activity spec §5.1, §5.2), which the log rows use.
mod activity_words {
    use super::*;

    /// Spec §5.1 with the box clause: the user's sentence verbatim when the named atom is near a
    /// face and the answer was box-affected; the other cases; rules 6–7 say "teaches it" only for
    /// a kept answer (plan review S13).
    #[test]
    fn the_box_clause_names_the_atom_at_the_edge() {
        use activity::words::{rule_sentence, RowCtx};
        use sim::activity::{TeachRule as R, TrustKey};
        let cx = |box_affected, near_face, kept| RowCtx { box_affected, near_face, kept };
        assert_eq!(
            rule_sentence(R::Spread, Some(0), cx(true, true, false)),
            "atom 0 is outside what the model has learned and is touching the box edge — move it inward"
        );
        assert_eq!(
            rule_sentence(R::Grade, Some(2), cx(true, false, false)),
            "atom 2 is outside what the model has learned — this answer also touches the box walls, so it is not learned here; a bigger cell would help"
        );
        assert_eq!(
            rule_sentence(R::Grade, Some(2), cx(true, false, true)),
            "atom 2 is outside what the model has learned — this answer also touches the box walls, so it stays in the fit only until enough others exist; a bigger cell would help"
        );
        assert_eq!(rule_sentence(R::Grade, Some(2), cx(false, true, true)), "atom 2 is outside what the model has learned");
        assert_eq!(rule_sentence(R::ForceUncertainty, Some(1), cx(false, false, true)), "the model is unsure here (atom 1) — this answer teaches it");
        assert_eq!(rule_sentence(R::ForceUncertainty, Some(1), cx(false, false, false)), "the model is unsure here (atom 1)");
        assert_eq!(
            rule_sentence(R::ForceUncertainty, Some(1), cx(true, true, false)),
            "the model is unsure here (atom 1), and atom 1 is touching the box edge — move it inward"
        );
        assert_eq!(
            rule_sentence(R::NewSpecies(8), None, RowCtx::default()),
            "the model has never seen O — answers like this one teach it; it can answer for O after a refit with O's reference energy and enough O samples"
        );
        assert_eq!(rule_sentence(R::Untrusted(TrustKey::Samples), None, RowCtx::default()), "the model was not trusted then: too few samples in the fit");
    }

    /// Short labels as the user wrote them (spec §3.3.3); 5d and 5e at γ = ∞ are different tests
    /// (plan review S7).
    #[test]
    fn rule_labels_are_the_codes_the_user_reads() {
        use activity::words::rule_label;
        use sim::activity::{PassJob, TeachRule as R};
        assert_eq!(rule_label(R::Grade, None), "5e grade");
        assert_eq!(rule_label(R::UnspannedAtom, None), "5e new environment");
        assert_eq!(rule_label(R::NewEnvironment, None), "5d new environment");
        assert_eq!(rule_label(R::IsolatedAtom, None), "5a lone atom");
        assert_eq!(rule_label(R::Cell, None), "0 cell");
        assert_eq!(rule_label(R::Pass(PassJob::HotBurst), Some(1830.0)), "hot burst 1830 K");
        assert_eq!(rule_label(R::Pass(PassJob::Reference { z: 1, multiplicity: 2 }), None), "reference H m2");
        // an audit row carries its verdict
        let mut e = sim::TeachEvent::new(sim::activity::TeachOrigin::Audit, R::Audit, sim::activity::TeachOutcome::Kept);
        e.audit = Some(sim::AuditVerdict::Failed);
        assert_eq!(activity::log::rule_cell(&e), "8 audit (failed)");
    }

    /// Plan review S13: the outcome words follow the policy's constants and say what happened.
    #[test]
    fn outcome_words_follow_the_constants_and_the_facts() {
        use activity::words::{outcome_label, outcome_sentence};
        use providers::surrogate::{FENCE_VOTES, THIN_EVERY};
        use sim::activity::TeachOutcome as K;
        assert_eq!(outcome_sentence(K::Thinned), format!("skipped: too similar to a sample already held (one in {THIN_EVERY} such is kept)"));
        assert!(outcome_sentence(K::KeptByVote).starts_with(&format!("kept: {FENCE_VOTES} fresh answers agree with each other")));
        assert_eq!(outcome_label(K::NotOffered), "not kept: other settings");
        assert_eq!(outcome_sentence(K::NotOffered), "used for this step, not kept for this model: its settings differ");
        assert_eq!(outcome_label(K::Discarded), "discarded");
    }

    /// Every rule and outcome has a label and a sentence (an exhaustive match stops compiling
    /// when a variant is added; this keeps the words non-empty).
    #[test]
    fn every_rule_and_outcome_is_worded() {
        use activity::words::{outcome_label, outcome_sentence, rule_label, rule_sentence, RowCtx};
        for r in [sim::activity::TeachRule::NonFiniteAnswer, sim::activity::TeachRule::NearQuarantine, sim::activity::TeachRule::Audit, sim::activity::TeachRule::RelaxVerification] {
            assert!(!rule_label(r, None).is_empty() && !rule_sentence(r, None, RowCtx::default()).is_empty(), "{r:?}");
        }
        for o in [sim::activity::TeachOutcome::Failed, sim::activity::TeachOutcome::NotSolved, sim::activity::TeachOutcome::KeptReleased { released: 2, superseded: 1 }] {
            assert!(!outcome_label(o).is_empty() && !outcome_sentence(o).is_empty(), "{o:?}");
        }
    }
}

/// The status chip's headline and the toolbar's progress bar (activity spec §3.1, §3.2).
mod activity_headline {
    use super::*;

    /// Spec §3.1: one headline per state, first match wins.
    #[test]
    fn the_headline_names_what_runs() {
        use activity::headline::{headline, BarMode};
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let running = |step| sim::ProbeView { main: sim::activity::SlotView { running: true, scf_step: step, elapsed_s: 0.4, solves: 1, ..Default::default() }, ..Default::default() };
        // a DFT step on the GPU backend
        let mut snap = snapshot(&scene, Backend::Gpu, None);
        snap.dynamics.eval.scf_steps = 11;
        let p = running(7);
        let inp = PanelInput { busy: true, probe: Some(p), ..input(&scene, Some(&snap), &cam) };
        let h = headline(&inp, inp.probe.as_ref()).unwrap();
        assert_eq!(h.text, "Solving · iteration 7 (last solve 11)");
        assert_eq!(h.bar, BarMode::Sweep);
        // estimated: a fallback
        let snap = snapshot(&scene, Backend::Estimated, Some(model()));
        let inp = PanelInput { busy: true, playing: true, probe: Some(running(3)), ..input(&scene, Some(&snap), &cam) };
        assert!(headline(&inp, inp.probe.as_ref()).unwrap().text.starts_with("DFT fallback · iteration 3"));
        // a pass with its ETA
        let mut snap = snapshot(&scene, Backend::Estimated, Some(crate::sim::ModelStatus { pass_eta_s: Some(100.0), ..model() }));
        snap.dynamics.sampling = Some(sim::SamplingProgress { done: 41, total: 273, stage: "scans", last_energy: -1.1, cancelled: false, skipped: 0, failed: 0, finished: false, target: None });
        let inp = PanelInput { busy: true, probe: Some(sim::ProbeView::default()), ..input(&scene, Some(&snap), &cam) };
        let h = headline(&inp, inp.probe.as_ref()).unwrap();
        assert_eq!(h.text, "Sample 42 of 273 · bond scan · 1:40 left");
        assert!(matches!(h.bar, BarMode::Fraction(f) if (f - 41.0 / 273.0).abs() < 1e-6));
        // a refit
        let snap = snapshot(&scene, Backend::Estimated, Some(model()));
        let inp = PanelInput { busy: true, probe: Some(sim::ProbeView { refit: Some((0.1, 250)), ..Default::default() }), ..input(&scene, Some(&snap), &cam) };
        assert_eq!(headline(&inp, inp.probe.as_ref()).unwrap().text, "Refitting · 250 samples");
        // the CPU solver (no step count)
        let mut snap = snapshot(&scene, Backend::Cpu, None);
        snap.engine = Backend::Cpu;
        let p = sim::ProbeView { main: sim::activity::SlotView { running: true, scf_step: 0, elapsed_s: 12.3, solves: 0, ..Default::default() }, ..Default::default() };
        let inp = PanelInput { busy: true, probe: Some(p), ..input(&scene, Some(&snap), &cam) };
        assert_eq!(headline(&inp, inp.probe.as_ref()).unwrap().text, "Solving (CPU reference) · 12 s");
        // a background audit beside a model-answered run: a second pill
        let snap = snapshot(&scene, Backend::Estimated, Some(model())); // audit_in_flight: true
        let p = sim::ProbeView { side: sim::activity::SlotView { running: true, scf_step: 4, elapsed_s: 0.1, solves: 2, ..Default::default() }, ..Default::default() };
        let inp = PanelInput { busy: true, playing: true, probe: Some(p), ..input(&scene, Some(&snap), &cam) };
        let h = headline(&inp, inp.probe.as_ref()).unwrap();
        assert_eq!(h.text, "playing");
        assert_eq!(h.audit.as_deref(), Some("Audit running · iteration 4"));
        // idle: no headline (the old idle texts stay)
        let snap = snapshot(&scene, Backend::Gpu, None);
        let inp = PanelInput { probe: Some(sim::ProbeView::default()), ..input(&scene, Some(&snap), &cam) };
        assert!(headline(&inp, inp.probe.as_ref()).is_none());
        assert_eq!(overlays::state(&inp).1, "paused — converged");
    }

    /// The orbital viewer's "computing orbitals" shares the chip: it is the headline when nothing
    /// else runs, and a pill beside the solver's when something does. Bond references alone give
    /// the bar a fraction and leave the chip its old state text.
    #[test]
    fn the_chip_handles_the_orbital_viewer_and_the_bond_references() {
        use activity::headline::{headline, BarMode};
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let snap = snapshot(&scene, Backend::Gpu, None);
        let idle = sim::ProbeView::default();
        let inp = PanelInput { orbital_waiting: true, probe: Some(idle), ..input(&scene, Some(&snap), &cam) };
        let h = headline(&inp, inp.probe.as_ref()).unwrap();
        assert_eq!((h.text.as_str(), h.orbitals.as_deref()), ("Computing orbitals…", None));
        let running = sim::ProbeView { main: sim::activity::SlotView { running: true, scf_step: 5, elapsed_s: 0.2, solves: 1, ..Default::default() }, ..Default::default() };
        let inp = PanelInput { orbital_waiting: true, busy: true, probe: Some(running), ..input(&scene, Some(&snap), &cam) };
        let h = headline(&inp, inp.probe.as_ref()).unwrap();
        assert!(h.text.starts_with("Solving · iteration 5") && h.orbitals.as_deref() == Some("Computing orbitals"), "{h:?}");
        let mut refs = snapshot(&scene, Backend::Gpu, None);
        refs.dynamics.bond_reference = Some(sim::ReferenceProgress { z: 8, done: 1, total: 3, elapsed_s: 14.2 });
        let inp = PanelInput { probe: Some(idle), ..input(&scene, Some(&refs), &cam) };
        let h = headline(&inp, inp.probe.as_ref()).unwrap();
        assert!(h.text.is_empty() && matches!(h.bar, BarMode::Fraction(f) if (f - 1.0 / 3.0).abs() < 1e-6), "{h:?}");
    }

    /// While the pass is in its references stage the text names the last reference that *finished*
    /// (a reference row lands when its solve completes), from this pass's rows only (activity final
    /// review I2): two species, then a second pass whose references have not landed yet.
    #[test]
    fn the_reference_stage_names_the_last_finished_species_of_this_pass() {
        use sim::activity::{PassJob, TeachOrigin as O, TeachOutcome as K, TeachRule as R};
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let sampling = |done, stage| sim::SamplingProgress { done, total: 273, stage, last_energy: -1.1, cancelled: false, skipped: 0, failed: 0, finished: false, target: None };
        let reference = |z| activity_tab::event(O::Pass, R::Pass(PassJob::Reference { z, multiplicity: 1 }), K::Reference, None);
        let mut st = activity::ActivityState::new();
        let mut s = snapshot(&scene, Backend::Estimated, Some(ModelStatus { pass_eta_s: None, ..model() }));
        let probe = sim::ProbeView { main: sim::activity::SlotView { running: true, scf_step: 9, elapsed_s: 0.2, solves: 1, ..Default::default() }, ..Default::default() };
        let text = |s: &Snapshot, reference| {
            let inp = PanelInput { busy: true, probe: Some(probe), ..input(&scene, Some(s), &cam) };
            activity::headline::display(&inp, reference).unwrap().text
        };
        // the pass starts: nothing has finished yet, so no species is named
        s.dynamics.sampling = Some(sampling(0, "references"));
        st.absorb(&s);
        assert_eq!(st.current_reference(), None);
        assert_eq!(text(&s, st.current_reference()), "Sample 1 of 273 · reference · iteration 9");
        // H finishes: it is the last done while O is solved
        std::thread::sleep(Duration::from_millis(3));
        s.activity.events = vec![reference(1)];
        s.activity.run_started = None;
        s.dynamics.sampling = Some(sampling(1, "references"));
        st.absorb(&s);
        assert_eq!(st.current_reference(), Some(1));
        assert_eq!(text(&s, st.current_reference()), "Sample 2 of 273 · reference (last done: H) · iteration 9");
        // O finishes: the newest of two
        std::thread::sleep(Duration::from_millis(3));
        s.activity.events = vec![reference(8)];
        s.dynamics.sampling = Some(sampling(2, "references"));
        st.absorb(&s);
        assert_eq!(st.current_reference(), Some(8));
        // other stages are left alone
        s.dynamics.sampling = Some(sampling(30, "scans"));
        assert_eq!(text(&s, Some(8)), "Sample 31 of 273 · bond scan · iteration 9");
        // the pass ends, a second pass starts on another scene: the old pass's rows do not count
        s.activity.events = Vec::new();
        s.dynamics.sampling.as_mut().unwrap().finished = true;
        st.absorb(&s);
        assert_eq!(st.current_reference(), None, "no pass running");
        std::thread::sleep(Duration::from_millis(3));
        s.dynamics.sampling = Some(sampling(0, "references"));
        st.absorb(&s);
        assert_eq!(st.current_reference(), None, "the first pass's O row is not this pass's");
        assert_eq!(text(&s, st.current_reference()), "Sample 1 of 273 · reference · iteration 9");
    }

    /// The solver's own iteration record (the probe's `live`): "iteration 37 / 200" of the cap, the
    /// cold retry named, and the pass's "Sample n of N" with it; the job row carries the record for
    /// its detail line (Δρ → target, ΔE, ms per iteration) and the stall tag once the best Δρ has
    /// not halved for `STALL_TAG_FROM` iterations.
    #[test]
    fn the_headline_and_the_job_row_show_the_solvers_iteration() {
        use activity::headline::headline;
        use activity::jobs::{rows, scf_lines, JobKey, STALL_TAG_FROM};
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let record = |it: u32, attempt: u32, since: u32| sim::ScfLive {
            attempt,
            iteration: it,
            max_iter: 200,
            delta_rho: 3.2e-5,
            tol: 1e-6,
            energy: -17.4,
            d_energy: -2.1e-4,
            ms_per_iteration: 85.4,
            since_gain: since,
            window: 50,
            best_delta_rho: 3.2e-2,
            stall_level: 8e-3,
            spark: [-2.0; sim::activity::SPARK_POINTS],
            spark_len: 3,
            spark_stride: 1,
        };
        let probe = |l| sim::ProbeView { main: sim::activity::SlotView { running: true, scf_step: 41, elapsed_s: 3.0, solves: 4, live: Some(l) }, ..Default::default() };
        let mut snap = snapshot(&scene, Backend::Estimated, Some(ModelStatus { pass_eta_s: None, audit_in_flight: false, ..model() }));
        snap.dynamics.sampling = Some(sim::SamplingProgress { done: 41, total: 158, stage: "scans", last_energy: -1.1, cancelled: false, skipped: 0, failed: 0, finished: false, target: None });
        let inp = PanelInput { busy: true, probe: Some(probe(record(37, 1, 3))), ..input(&scene, Some(&snap), &cam) };
        assert_eq!(headline(&inp, inp.probe.as_ref()).unwrap().text, "Sample 42 of 158 · bond scan · iteration 37 / 200");
        let r = rows(&inp, inp.probe.as_ref());
        let pass = r.iter().find(|r| r.key == JobKey::Pass).unwrap();
        assert_eq!(pass.detail, "bond scan · sample 42 of 158 · iteration 37 / 200");
        let l = pass.scf.expect("the pass row carries the record");
        assert_eq!(scf_lines(&l), ("Δρ 3.2e-5 → 1e-6 · ΔE −2.1e-4 Ha · 85 ms/iteration".to_string(), None));
        // the stall countdown, from STALL_TAG_FROM on
        assert_eq!(scf_lines(&record(80, 1, STALL_TAG_FROM - 1)).1, None);
        assert_eq!(scf_lines(&record(80, 1, 31)).1.as_deref(), Some("no 2× gain for 31 / 50"));
        // ruling (a): below the stall level the stop does not apply, so there is no countdown to show
        assert_eq!(scf_lines(&sim::ScfLive { best_delta_rho: 4.8e-5, stall_level: 2e-3, ..record(80, 1, 31) }).1, None);
        // the providers' cold retry counts from 1 again and says so
        let inp = PanelInput { busy: true, probe: Some(probe(record(12, 2, 0))), ..input(&scene, Some(&snap), &cam) };
        assert_eq!(headline(&inp, inp.probe.as_ref()).unwrap().text, "Sample 42 of 158 · bond scan · iteration 12 / 200 · cold retry");
        // outside a pass: a DFT solve's iteration, with the last solve's count
        let mut gpu = snapshot(&scene, Backend::Gpu, None);
        gpu.dynamics.eval.scf_steps = 11;
        let inp = PanelInput { busy: true, probe: Some(probe(record(7, 1, 0))), ..input(&scene, Some(&gpu), &cam) };
        assert_eq!(headline(&inp, inp.probe.as_ref()).unwrap().text, "Solving · iteration 7 / 200 (last solve 11)");
        let r = rows(&inp, inp.probe.as_ref());
        assert_eq!(r.iter().find(|r| r.key == JobKey::LiveSolve).unwrap().detail, "iteration 7 / 200 (last solve 11)");
        // the CPU solver reports its iterations too (its observer): no "CPU reference" fallback text then
        let mut cpu = snapshot(&scene, Backend::Cpu, None);
        cpu.engine = Backend::Cpu;
        let p = sim::ProbeView { main: sim::activity::SlotView { running: true, scf_step: 0, elapsed_s: 9.0, solves: 0, live: Some(record(4, 1, 0)) }, ..Default::default() };
        let inp = PanelInput { busy: true, probe: Some(p), ..input(&scene, Some(&cpu), &cam) };
        assert_eq!(headline(&inp, inp.probe.as_ref()).unwrap().text, "Solving · iteration 4 / 200");
    }

    /// User request 2026-10-17: between two samples of a pass (the probe idle, its last solve's record still held) the
    /// pass row keeps that record, so the iteration block stays on screen instead of blinking out; a new solve's
    /// `busy` clears it until its first iteration (the row then reserves the block's height).
    #[test]
    fn the_pass_row_keeps_the_last_record_between_samples() {
        use activity::jobs::{rows, JobKey};
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let mut snap = snapshot(&scene, Backend::Estimated, Some(ModelStatus { pass_eta_s: None, audit_in_flight: false, ..model() }));
        snap.dynamics.sampling = Some(sim::SamplingProgress { done: 7, total: 158, stage: "scans", last_energy: -1.1, cancelled: false, skipped: 0, failed: 0, finished: false, target: None });
        let probe = sim::ActivityProbe::new();
        let busy = probe.busy(sim::Slot::Main);
        probe.scf_iteration(sim::Slot::Main, &sim::ScfIteration { iteration: 14, max_iter: 200, delta_rho: 8e-7, tol: 1e-6, energy: -17.4, since_gain: 0, window: 50, best_delta_rho: 8e-7, stall_level: 8e-3 });
        drop(busy); // the sample finished; the next has not started
        let idle = probe.view();
        assert!(!idle.main.running);
        let inp = PanelInput { busy: true, probe: Some(idle), ..input(&scene, Some(&snap), &cam) };
        let r = rows(&inp, inp.probe.as_ref());
        let pass = r.iter().find(|r| r.key == JobKey::Pass).unwrap();
        assert_eq!(pass.scf.map(|l| (l.iteration, l.delta_rho)), Some((14, 8e-7)), "the last sample's record stays");
        assert!(!pass.detail.contains("iteration"), "the detail names no running iteration: {}", pass.detail);
        // the next sample starts: the record is cleared until its first iteration
        let _next = probe.busy(sim::Slot::Main);
        let inp = PanelInput { busy: true, probe: Some(probe.view()), ..input(&scene, Some(&snap), &cam) };
        assert!(rows(&inp, inp.probe.as_ref()).iter().find(|r| r.key == JobKey::Pass).unwrap().scf.is_none());
    }

    /// The solve log's iteration cell: the count and how the SCF ended.
    #[test]
    fn the_solve_log_shows_iterations_and_the_end() {
        use activity::log::{solve_cell, solve_hover};
        use sim::activity::{PassJob, TeachOrigin as O, TeachOutcome as K, TeachRule as R};
        let mut e = sim::TeachEvent::new(O::Pass, R::Pass(PassJob::Scan), K::Kept);
        e.solve_ms = 840.0;
        e.scf_steps = Some(37);
        e.scf_end = Some(sim::ScfEnd::Converged);
        assert_eq!(solve_cell(&e), "840 ms · 37 it");
        assert_eq!(solve_hover(&e).as_deref(), Some("37 SCF iterations — converged"));
        e.outcome = K::Failed;
        e.solve_ms = 12_100.0;
        e.scf_steps = Some(61);
        e.scf_end = Some(sim::ScfEnd::Stalled);
        assert_eq!(solve_cell(&e), "12.1 s · 61 stalled");
        assert!(solve_hover(&e).unwrap().starts_with("61 SCF iterations — stalled"));
        e.scf_steps = Some(200);
        e.scf_end = Some(sim::ScfEnd::Cap);
        assert_eq!(solve_cell(&e), "12.1 s · 200 cap");
        // a solve without an SCF count, and a row with no solve
        e.scf_steps = None;
        e.scf_end = None;
        assert_eq!((solve_cell(&e), solve_hover(&e)), ("12.1 s".to_string(), None));
        e.outcome = K::NotSolved;
        assert_eq!(solve_cell(&e), "—");
    }

    /// Spec §6: the headline is rebuilt at most every 100 ms.
    #[test]
    fn the_headline_refreshes_at_most_every_100_ms() {
        use activity::headline::{Throttled, REFRESH};
        let t0 = std::time::Instant::now();
        let mut t = Throttled::new(0u32);
        assert_eq!(*t.get(t0, || 1), 1);
        assert_eq!(*t.get(t0 + REFRESH / 2, || 2), 1, "inside 100 ms: cached");
        assert_eq!(*t.get(t0 + REFRESH, || 3), 3);
    }

    /// Review Focus 3: a long solve keeps its headline, and its elapsed time moves.
    #[test]
    fn a_stale_probe_is_not_a_headline() {
        use activity::headline::headline;
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let mut snap = snapshot(&scene, Backend::Cpu, None);
        snap.engine = Backend::Cpu;
        let p = sim::ProbeView { main: sim::activity::SlotView { running: true, scf_step: 0, elapsed_s: 31.0, solves: 0, ..Default::default() }, ..Default::default() };
        let inp = PanelInput { busy: true, probe: Some(p), ..input(&scene, Some(&snap), &cam) };
        assert_eq!(headline(&inp, inp.probe.as_ref()).unwrap().text, "Solving (CPU reference) · 31 s");
        let p = sim::ProbeView::default();
        let inp = PanelInput { busy: false, probe: Some(p), ..input(&scene, Some(&snap), &cam) };
        assert!(headline(&inp, inp.probe.as_ref()).is_none(), "a finished solve leaves no headline");
    }

    /// The bar draws on the toolbar's top edge while work runs, and a click on it opens the
    /// Activity tab; idle, there is no bar to click.
    #[test]
    fn the_progress_bar_opens_the_activity_tab() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let snap = snapshot(&scene, Backend::Gpu, None);
        let probe = sim::ProbeView { main: sim::activity::SlotView { running: true, scf_step: 5, elapsed_s: 0.2, solves: 1, ..Default::default() }, ..Default::default() };
        let inp = PanelInput { busy: true, probe: Some(probe), ..input(&scene, Some(&snap), &cam) };
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        st.left_open = false;
        click_widget(&ctx, &mut st, &inp, "activity-bar");
        assert!(st.tab_shown(DockTab::Activity), "the bar opens the left dock on Activity");
        // idle: nothing is drawn, so nothing is probed
        PROBES.with(|p| p.borrow_mut().clear());
        let idle = input(&scene, Some(&snap), &cam);
        let mut st = UiState::new("h2", &scene);
        for _ in 0..3 {
            frame(&ctx, &mut st, &idle, 1400.0, vec![]);
        }
        assert!(PROBES.with(|p| !p.borrow().contains_key("activity-bar")));
    }
}

/// Plain wording and the live trust checklist (activity spec §3.3.4, §5.3).
mod activity_trust {
    use super::*;

    /// Plan review N7: mHa with two significant figures (σ_f 4.5e-5 Ha/Bohr is "0.045", not "0.04").
    #[test]
    fn mha_keeps_two_significant_figures() {
        use activity::words::mha;
        assert_eq!(mha(1e-3), "1");
        assert_eq!(mha(2.2e-3), "2.2");
        assert_eq!(mha(4.5e-5), "0.045");
        assert_eq!(mha(4.5e-4), "0.45");
        assert_eq!(mha(1.23e-2), "12");
        assert_eq!(mha(0.0), "0");
    }

    /// Plan review M1 and the Task 3 review: the audit line is per atom or names forces, and when
    /// met it says what the last audit was — or that none ran this fit, or ever — never "passed"
    /// by default.
    #[test]
    fn the_audit_line_says_what_ran() {
        use activity::words::{trust_line, AuditSeen};
        use sim::activity::{TrustCheck, TrustKey as K};
        use sim::AuditVerdict as V;
        let met = TrustCheck { key: K::Audit, met: true, value: 0.0, limit: 1e-3 };
        let rec = |verdict| sim::AuditRecord { gap: 2e-4, verdict, face_density: None, unaudited_steps: 0, asynchronous: true, teacher_ms: 90.0 };
        let head = "Last check against DFT — no failed check since the last refit";
        assert_eq!(trust_line(&met, AuditSeen::Never, 20), format!("{head} (last check: none yet, one every 20 answered steps)"));
        assert_eq!(trust_line(&met, AuditSeen::Never, 0), format!("{head} (last check: none, audits are off)"));
        assert_eq!(trust_line(&met, AuditSeen::ThisFit(&rec(V::Passed)), 20), format!("{head} (last check: passed)"));
        assert_eq!(trust_line(&met, AuditSeen::ThisFit(&rec(V::BoxAffected)), 20), format!("{head} (last check: box-affected, not judged)"));
        // a refit since the last audit: not "passed", not "none yet"
        assert_eq!(
            trust_line(&met, AuditSeen::EarlierFit(&rec(V::Passed)), 20),
            format!("{head} (last check: none since the last refit — the fit before it was last checked: passed)")
        );
        assert!(trust_line(&met, AuditSeen::EarlierFit(&rec(V::Failed)), 20).contains("the fit before it was last checked: failed"));
        let energy = TrustCheck { key: K::Audit, met: false, value: 2.5e-3, limit: 1e-3 };
        assert_eq!(trust_line(&energy, AuditSeen::ThisFit(&rec(V::Failed)), 20), "Last check against DFT — found the energy off by 2.5 mHa/atom, allowed 1 mHa/atom (until a refit predicts it without that sample)");
        let forces = TrustCheck { key: K::AuditForces, met: false, value: 4.2e-3, limit: 2e-3 };
        assert_eq!(trust_line(&forces, AuditSeen::ThisFit(&rec(V::Failed)), 20), "Last check against DFT — found forces off by 4.2, allowed 2 mHa/Bohr (until a refit predicts it without that sample)");
        // the three states come from the status
        let m = model();
        assert!(matches!(activity::words::AuditSeen::of(&m), AuditSeen::ThisFit(_)));
        assert!(matches!(activity::words::AuditSeen::of(&ModelStatus { last_audit_this_fit: false, ..model() }), AuditSeen::EarlierFit(_)));
        assert!(matches!(activity::words::AuditSeen::of(&ModelStatus { last_audit: None, last_audit_this_fit: false, ..model() }), AuditSeen::Never));
    }

    /// Plan review M3: the class line words the class's own verdict, so an energy-only failure
    /// is an energy error.
    #[test]
    fn the_class_line_words_the_failing_condition() {
        use activity::words::class_line;
        use sim::{ClassKey, ClassLoo};
        let l = |class, refusal: Option<(ClassKey, f64, f64)>| ClassLoo { class, samples: 120, force_rms: 0.1, force_max: 0.4, energy_max: 1100.0, trusted: refusal.is_none(), refusal };
        assert_eq!(class_line(&l("pair", None)), "Two-atom arrangements — learned");
        assert_eq!(class_line(&l("pair", Some((ClassKey::Energy, 1100.0, 1.0)))), "Two-atom arrangements — not yet (leave-one-out energy error 1100.0× the tier)");
        assert_eq!(class_line(&l("pair", Some((ClassKey::Force, 2.69, 1.0)))), "Two-atom arrangements — not yet (leave-one-out force error 2.7× the bound)");
        assert_eq!(class_line(&l("many-body", Some((ClassKey::Samples, 12.0, 40.0)))), "Three-or-more-atom arrangements — not yet (12 < 40 samples)");
    }

    /// Review Focus 4 on the diag-3h fixture: the box line is first and "this is why"; the fit's own
    /// failed condition is ✗ below it with its number.
    #[test]
    fn the_three_hydrogen_refusal_names_the_box_first() {
        use sim::activity::{TrustCheck, TrustKey};
        let reason = "max training energy residual 2.2e-3 > 1.0e-3 Ha/atom";
        let m = ModelStatus {
            trusted: false,
            untrusted_reason: Some(reason.into()),
            samples: 145,
            quarantined: 0,
            box_affected: 130,
            box_excluded: 0,
            trust_floor: Some(59),
            trust_checks: vec![
                TrustCheck { key: TrustKey::Samples, met: true, value: 145.0, limit: 29.0 },
                TrustCheck { key: TrustKey::ResidualDofE, met: true, value: 120.0, limit: 30.0 },
                TrustCheck { key: TrustKey::ResidualDofF, met: true, value: 700.0, limit: 30.0 },
                TrustCheck { key: TrustKey::EnergyResidual, met: false, value: 2.2e-3, limit: 1e-3 },
                TrustCheck { key: TrustKey::ForceResidual, met: true, value: 1e-3, limit: 2e-3 },
                TrustCheck { key: TrustKey::ForceNoise, met: true, value: 4e-4, limit: 2e-3 },
                TrustCheck { key: TrustKey::Audit, met: true, value: 0.0, limit: 1e-3 },
            ],
            ..model()
        };
        let scene = Scene::preset("h2").unwrap();
        let st = UiState::new("h2", &scene);
        let lines = activity::trust::lines(&m, &scene, &st);
        assert!(lines[0].why && !lines[0].met && lines[0].text.starts_with("Samples clear of the box walls — 130 in the fit touch them (145 held)"), "{:?}", lines[0]);
        let e = lines.iter().find(|l| l.text.starts_with("Reproduces its own training energies")).unwrap();
        assert!(!e.met && !e.why && e.text.contains("worst 2.2 mHa/atom, allowed 1 mHa/atom"));
        assert_eq!(lines.iter().filter(|l| l.why).count(), 1);
        // the Model tab's line uses the box sentence, the old reason on hover
        let t = trust(Some(&m), &scene, &st);
        assert!(t.text.starts_with("no — Samples clear of the box walls") && t.detail.as_deref().is_some_and(|d| d.contains(reason)));
    }

    /// Plan review S15: the box is blamed only when the first unmet check is a condition of the
    /// fit it is in. Not a count (Sample more cures it), not a failed audit (judged against a
    /// fresh teacher answer that is not box-affected).
    #[test]
    fn the_box_is_not_blamed_for_a_failed_audit_or_a_count() {
        use sim::activity::TrustKey as K;
        let m = ModelStatus { trusted: false, untrusted_reason: Some("…".into()), samples: 145, quarantined: 0, box_affected: 130, box_excluded: 0, trust_floor: Some(59), ..model() };
        for key in [K::EnergyResidual, K::ForceResidual, K::ForceNoise] {
            assert_eq!(crate::sim::box_blocks_trust(&ModelStatus { trust_checks: checks_failing(key), ..m.clone() }), Some((130, 145)), "{key:?}");
        }
        for key in [K::Samples, K::ResidualDofE, K::ResidualDofF, K::Audit, K::AuditForces] {
            assert_eq!(crate::sim::box_blocks_trust(&ModelStatus { trust_checks: checks_failing(key), ..m.clone() }), None, "{key:?}");
        }
        // without a check list nothing says which condition failed: no blame
        assert_eq!(crate::sim::box_blocks_trust(&m), None);
    }

    /// Without the box, the first ✗ is "this is why" and the Model tab says it in plain words.
    #[test]
    fn the_first_unmet_check_is_this_is_why() {
        use sim::activity::{TrustCheck, TrustKey};
        let m = ModelStatus {
            trusted: false,
            untrusted_reason: Some("n_samples 12 < q_prime 16".into()),
            trust_checks: vec![TrustCheck { key: TrustKey::Samples, met: false, value: 12.0, limit: 16.0 }],
            ..model()
        };
        let scene = Scene::preset("h2").unwrap();
        let st = UiState::new("h2", &scene);
        let lines = activity::trust::lines(&m, &scene, &st);
        assert!(lines[0].why && lines[0].text == "Enough samples in the fit — 12 of 16 needed (one per independent direction the model can bend)");
        assert_eq!(trust(Some(&m), &scene, &st).text, "no — Enough samples in the fit — 12 of 16 needed (one per independent direction the model can bend)");
        // met: every line ✓, none "this is why", the box line ✓, the class lines after
        let ok = ModelStatus { trusted: true, untrusted_reason: None, trust_checks: vec![TrustCheck { key: TrustKey::Samples, met: true, value: 145.0, limit: 16.0 }], ..model() };
        let lines = activity::trust::lines(&ok, &scene, &st);
        assert!(lines.iter().all(|l| l.met && !l.why), "{lines:?}");
        assert!(lines.last().unwrap().text.starts_with("Two-atom arrangements"));
    }

    /// The stuck variant in plain words (spec §5.3); no stored reason, no invented rule (plan
    /// review S10).
    #[test]
    fn the_stuck_line_is_plain() {
        let info = sim::activity::StuckInfo { rule: Some(sim::activity::TeachRule::Spread), atom: Some(0), steps: 50, box_streak: true, gap: Some(2e-3), face_density: 4e-5 };
        let mut scene = Scene::preset("h2").unwrap();
        scene.nuclei[0].pos = [0.0, 0.0, 5.0]; // 1 Bohr from the +z face of the preset's 49³ × 0.25 cell (d_box(0.25) = 4.84)
        let sim_scene = scene.to_sim().unwrap();
        assert_eq!(
            activity::words::stuck_sentence(&info, &sim_scene),
            "trusted, but not answering here: atom 0 is outside what the model has learned and is touching the box edge — move it inward (the last 50 steps were DFT)"
        );
        let unknown = sim::activity::StuckInfo { rule: None, ..info };
        assert_eq!(activity::words::stuck_sentence(&unknown, &sim_scene), "trusted, but not answering here (the last 50 steps were DFT)");
        // the Model tab's line carries it, with the old string on hover
        let m = ModelStatus { stuck: Some("old text".into()), stuck_info: Some(info), ..model() };
        let st = UiState::new("h2", &scene);
        let t = trust(Some(&m), &scene, &st);
        assert!(t.text.starts_with("yes, but not answering here: atom 0 is outside what the model has learned and is touching the box edge"), "{}", t.text);
        assert_eq!(t.detail.as_deref(), Some("old text"));
    }

    /// The checklist draws on every state and the Model tab line keeps a hover with the fit's
    /// own words.
    #[test]
    fn the_checklist_draws_untrusted_and_trusted() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        st.backend = Backend::Estimated;
        st.open_tab(DockTab::Activity);
        let bad = ModelStatus { trusted: false, untrusted_reason: Some("x".into()), trust_checks: checks_failing(sim::activity::TrustKey::ForceResidual), ..model() };
        for m in [model(), bad] {
            let snap = snapshot(&scene, Backend::Estimated, Some(m));
            let inp = input(&scene, Some(&snap), &cam);
            for _ in 0..2 {
                assert!(frame(&ctx, &mut st, &inp, 1400.0, vec![]).commands.is_empty());
            }
        }
    }
}

/// The training plots, the timeline and their measured cost (activity spec §3.3.5, §3.3.6, §6).
mod activity_plots {
    use super::*;
    use std::sync::Arc;

    /// The plots' maths: log ranges, the band polygon (upper edge forward, lower edge back), the
    /// coordinate map; and each plot draws with no data, one point and full data.
    #[test]
    fn the_training_plots_draw_and_their_maths_holds() {
        use activity::plots::{band_polygon, log_range, to_screen};
        let (lo, hi) = log_range(&[1e-6, 1e-3, 0.0, f64::NAN]).unwrap();
        assert!((lo + 6.0).abs() < 1e-12 && (hi + 3.0).abs() < 1e-12, "zero and NaN are skipped");
        assert_eq!(log_range(&[]), None);
        let r = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(100.0, 50.0));
        let p = to_screen(r, (0.0, 1.0), (0.0, 1.0), 0.5, 0.5);
        assert_eq!((p.x, p.y), (50.0, 25.0));
        let poly = band_polygon(r, (0.0, 2.0), (-1.0, 1.0), &[0.0, 1.0, 2.0], &[0.0, 0.0, 0.0], &[0.1, 0.2, 0.1]);
        assert_eq!(poly.len(), 6);
        assert!(poly[0].y < poly[5].y, "upper edge above the lower one");
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let one_fit = vec![crate::sim::FitPoint { epoch: 1, samples: 40, rmse_e: 2e-5, rmse_f: 5e-5, trusted: true, at: Instant::now() }];
        let mut many_fits = one_fit.clone();
        for k in 0..30 {
            many_fits.push(crate::sim::FitPoint { epoch: 2 + k, samples: 50 + k as usize, rmse_e: 1e-3 / (1.0 + k as f64), rmse_f: 3e-3 / (1.0 + k as f64), trusted: k > 5, at: Instant::now() });
        }
        for (views, forces, fits, draft) in [
            (None, false, vec![], false),
            (None, false, one_fit.clone(), false),
            (Some(Arc::new(activity::fixture_views())), false, many_fits.clone(), false),
            (Some(Arc::new(activity::fixture_views())), true, many_fits.clone(), true),
        ] {
            let mut st = activity::ActivityState::new();
            st.views = views;
            st.fit_points = fits;
            st.training_open = true;
            st.parity_forces = forces;
            st.burst_hist = vec![0, 0, 3, 5, 9, 4, 1];
            let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 900.0))), ..Default::default() };
            let mut full = ctx.run_ui(raw, |ui| {
                // draft on in the force case, so both pair-curve captions draw
                activity::plots::training(ui, &mut st, Some(2000.0), draft);
                activity::plots::timeline(ui, &st);
            });
            full.textures_delta.clear();
        }
        // a curve without a posterior has neither band nor lane, and one with a mismatch says so
        let mut v = activity::fixture_views();
        v.curves[0].sigma.clear();
        v.curves[0].force_ratio.clear();
        let mut st = activity::ActivityState::new();
        st.views = Some(Arc::new(v));
        st.provenance_mismatch = true;
        let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 900.0))), ..Default::default() };
        let mut full = ctx.run_ui(raw, |ui| activity::plots::training(ui, &mut st, None, false));
        full.textures_delta.clear();
    }

    /// The timeline draws every kind of mark, the pass running and the pass cancelled.
    #[test]
    fn the_timeline_draws_every_mark() {
        use activity::Mark;
        use std::time::Duration as D;
        let mut st = activity::ActivityState::new();
        st.timeline = vec![
            Mark::Pass(D::from_millis(0), Some(D::from_millis(900)), false),
            Mark::Fallback(D::from_millis(1000)),
            Mark::Fallback(D::from_millis(1000)),
            Mark::Audit(D::from_millis(1100), sim::AuditVerdict::Passed),
            Mark::Audit(D::from_millis(1200), sim::AuditVerdict::Failed),
            Mark::Audit(D::from_millis(1300), sim::AuditVerdict::BoxAffected),
            Mark::Refit(D::from_millis(1400), true),
            Mark::Refit(D::from_millis(1450), false),
            Mark::Quarantine(D::from_millis(1500), 3),
            Mark::Pass(D::from_millis(1600), None, false),
            Mark::Pass(D::from_millis(1700), Some(D::from_millis(1750)), true),
        ];
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 400.0))), ..Default::default() };
        let mut full = ctx.run_ui(raw, |ui| activity::plots::timeline(ui, &st));
        full.textures_delta.clear();
    }

    /// Plan review M5: the curve's refusal styling comes from the policy's verdict (`refused`,
    /// from `would_answer`), stretch by stretch with its rule; the y range from the answered points
    /// and the dots only, so a refused extrapolation toward 0.5 Bohr cannot set the scale; the lane
    /// compares σ_epi/N with the tier (rule 7) as a ratio, beside rule 6's ratio.
    #[test]
    fn the_pair_curve_reads_the_policy_verdict() {
        use activity::plots::{curve_y_range, lane_ratios, refused_stretches};
        use sim::activity::TeachRule as R;
        let refused = [Some(R::IsolatedAtom), Some(R::IsolatedAtom), None, None, Some(R::Grade), Some(R::IsolatedAtom)];
        assert_eq!(refused_stretches(&refused), vec![(0, 2, R::IsolatedAtom), (4, 5, R::Grade), (5, 6, R::IsolatedAtom)]);
        assert!(refused_stretches(&[None, None]).is_empty());
        let mut c = activity::fixture_views().curves[0].clone();
        let n = c.r.len();
        c.energy = (0..n).map(|k| if k < 10 { 5.0 } else { -0.17 + 1e-3 * k as f64 }).collect();
        c.refused = (0..n).map(|k| (k < 10).then_some(R::Coordination)).collect();
        c.dots = vec![(1.4, -0.2, sim::activity::SampleState::Fitted)];
        let (lo, hi) = curve_y_range(&c).unwrap();
        assert!(hi < 1.0, "the refused wall at short r does not set the scale: {hi}");
        assert!((lo + 0.2).abs() < 1e-12 && (hi - (-0.17 + 1e-3 * (n - 1) as f64)).abs() < 1e-12, "answered points and dots: {lo} {hi}");
        c.sigma = vec![2e-3; n];
        let (energy, floor) = lane_ratios(&c, 4e-5, 1e-3);
        assert!((energy[0] - 1.0).abs() < 1e-12, "σ_epi/N ÷ tier with N = 2");
        assert!((floor - 0.02).abs() < 1e-12, "s_E ÷ tier: the per-dimer floor 2 s_E over N = 2");
        // every point refused: the whole curve sets the scale rather than nothing
        c.refused = vec![Some(R::Grade); n];
        c.dots.clear();
        assert!(curve_y_range(&c).is_some());
        c.energy.clear();
        assert_eq!(curve_y_range(&c), None);
    }

    /// The Activity tab with training open, on a state with views and fits, lays out in the dock
    /// and sends nothing.
    #[test]
    fn the_training_section_draws_in_the_dock() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        st.backend = Backend::Estimated;
        st.open_tab(DockTab::Activity);
        st.activity.training_open = true;
        st.activity.views_sent = true;
        let mut snap = snapshot(&scene, Backend::Estimated, Some(model()));
        snap.activity.events = activity_tab::every_kind_of_event();
        snap.activity.views = Some(Arc::new(activity::fixture_views()));
        snap.activity.activity_ms = 0.4;
        snap.activity.teacher_ms = 1200.0;
        snap.activity.fit_points = vec![crate::sim::FitPoint { epoch: 1, samples: 271, rmse_e: 2.1e-6, rmse_f: 1.6e-5, trusted: true, at: Instant::now() }];
        st.activity.absorb(&snap);
        let inp = input(&scene, Some(&snap), &cam);
        for w in [1000.0, 1400.0] {
            for _ in 0..2 {
                assert!(frame(&ctx, &mut st, &inp, w, vec![]).commands.is_empty());
            }
        }
        // the footer's ratio
        assert_eq!(st.activity.overhead, (0.4, 1200.0));
    }

    /// Spec §6: the Activity tab's frame cost with 5 000 rows and full views, printed. Never
    /// asserted (the ship criterion is read from the printout).
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn the_activity_tab_frame_cost_is_reported() {
        use sim::activity::{TeachOrigin as O, TeachOutcome as K, TeachRule as R};
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        st.backend = Backend::Estimated;
        st.open_tab(DockTab::Activity);
        st.activity.training_open = true;
        st.activity.views_sent = true;
        let mut snap = snapshot(&scene, Backend::Estimated, Some(model()));
        snap.activity.events = (0..5000).map(|i| activity_tab::event(O::Fallback, R::Grade, K::Kept, Some(i))).collect();
        let mut views = activity::fixture_views();
        // a trained model's views: 20 000 force components, 400 energy points
        views.parity_force = (0..20_000).map(|i| [(i as f32 * 0.37).sin() * 0.05, (i as f32 * 0.37).sin() * 0.05 + 1e-4]).collect();
        views.force_points_total = 20_000;
        views.parity_energy = (0..400)
            .map(|i| sim::activity::ParityPoint { dft: -0.1 - 1e-4 * i as f64, model: -0.1 - 1e-4 * i as f64 + 1e-5, kind: sim::activity::SampleKind::Pass, state: sim::activity::SampleState::Fitted, index: i })
            .collect();
        snap.activity.views = Some(Arc::new(views));
        snap.activity.fit_points = (0..200).map(|k| crate::sim::FitPoint { epoch: k, samples: 40 + k as usize, rmse_e: 1e-3 / (1.0 + k as f64), rmse_f: 1e-3 / (1.0 + k as f64), trusted: true, at: Instant::now() }).collect();
        st.activity.absorb(&snap);
        let inp = input(&scene, Some(&snap), &cam);
        // a frame as the window runs it: layout and shapes, then the tessellation the painter needs
        let one = |ctx: &egui::Context, st: &mut UiState| {
            let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0))), ..Default::default() };
            let mut full = ctx.run_ui(raw, |root| {
                show(root, st, &inp);
            });
            full.textures_delta.clear();
            let meshes = ctx.tessellate(full.shapes, full.pixels_per_point);
            std::hint::black_box(meshes.len());
        };
        let time = |ctx: &egui::Context, st: &mut UiState, tab: DockTab| {
            st.open_tab(tab);
            one(ctx, st);
            let t = std::time::Instant::now();
            for _ in 0..60 {
                one(ctx, st);
            }
            t.elapsed().as_secs_f64() * 1e3 / 60.0
        };
        let activity = time(&ctx, &mut st, DockTab::Activity);
        st.activity.training_open = false;
        st.activity.views_sent = false;
        let activity_closed = time(&ctx, &mut st, DockTab::Activity);
        let readouts = time(&ctx, &mut st, DockTab::Readouts);
        eprintln!(
            "Activity tab: {activity:.2} ms per frame with Training open (5 000 rows, 20 000 force points, 400 energy points, 1400 px; layout and tessellation), {activity_closed:.2} ms closed; the Readouts tab for comparison: {readouts:.2} ms"
        );
    }
}

/// Activity final review: the minor fixes that have a behaviour to pin.
mod activity_final {
    use super::*;
    use sim::activity::{TeachOrigin, TeachOutcome, TeachRule, TrustCheck, TrustKey};

    /// M3: a trusted model with box-affected samples in its fit does not get "clear of the box walls".
    #[test]
    fn the_met_box_line_says_what_is_true() {
        let scene = Scene::preset("h2").unwrap();
        let st = UiState::new("h2", &scene);
        let ok = |box_affected| ModelStatus {
            trusted: true,
            untrusted_reason: None,
            box_affected,
            box_excluded: 0,
            trust_checks: vec![TrustCheck { key: TrustKey::Samples, met: true, value: 145.0, limit: 16.0 }],
            ..model()
        };
        let text = |m: &ModelStatus| activity::trust::lines(m, &scene, &st).into_iter().map(|l| l.text).find(|t| t.to_lowercase().contains("box walls")).unwrap();
        assert_eq!(text(&ok(0)), "Samples clear of the box walls");
        assert_eq!(text(&ok(5)), "Box walls not blocking trust — 5 samples in the fit touch them (273 held)");
    }

    /// M1: no "iteration 0" for a CPU solve inside a pass, and a GPU solve before its first poll
    /// is not "CPU reference".
    #[test]
    fn the_job_rows_do_not_invent_a_step() {
        use activity::jobs::{rows, JobKey};
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let on = |step| sim::ProbeView { main: sim::activity::SlotView { running: true, scf_step: step, elapsed_s: 0.1, solves: 0, ..Default::default() }, ..Default::default() };
        let mut cpu = snapshot(&scene, Backend::Cpu, None);
        cpu.engine = Backend::Cpu;
        let inp = PanelInput { busy: true, probe: Some(on(0)), ..input(&scene, Some(&cpu), &cam) };
        let r = rows(&inp, inp.probe.as_ref());
        assert_eq!(r.iter().find(|r| r.key == JobKey::LiveSolve).unwrap().detail, "CPU reference: no iteration count yet");
        let gpu = snapshot(&scene, Backend::Gpu, None);
        let inp = PanelInput { busy: true, probe: Some(on(0)), ..input(&scene, Some(&gpu), &cam) };
        assert_eq!(rows(&inp, inp.probe.as_ref()).iter().find(|r| r.key == JobKey::LiveSolve).unwrap().detail, "starting");
        // a pass on the CPU teacher: the step stays out of the detail while it reads 0
        let mut pass = snapshot(&scene, Backend::Estimated, Some(model()));
        pass.dynamics.sampling = Some(sim::SamplingProgress { done: 3, total: 273, stage: "scans", last_energy: -1.0, cancelled: false, skipped: 0, failed: 0, finished: false, target: None });
        let inp = PanelInput { busy: true, probe: Some(on(0)), ..input(&scene, Some(&pass), &cam) };
        let r = rows(&inp, inp.probe.as_ref());
        assert!(!r.iter().find(|r| r.key == JobKey::Pass).unwrap().detail.contains("iteration"));
        let inp = PanelInput { busy: true, probe: Some(on(4)), ..input(&scene, Some(&pass), &cam) };
        assert!(rows(&inp, inp.probe.as_ref()).iter().find(|r| r.key == JobKey::Pass).unwrap().detail.contains("iteration 4"));
        assert_eq!(rows(&inp, inp.probe.as_ref()).iter().find(|r| r.key == JobKey::Pass).unwrap().detail, "bond scan · sample 4 of 273 · iteration 4");
    }

    /// M2: a worker switch closes an open pass as cancelled, forgets the quarantine baseline, and
    /// a repeated fit is not a second refit.
    #[test]
    fn a_new_worker_leaves_no_false_events() {
        use activity::Mark;
        let scene = Scene::preset("h2").unwrap();
        let mut st = activity::ActivityState::new();
        let mut s = snapshot(&scene, Backend::Estimated, Some(ModelStatus { quarantined: 0, ..model() }));
        s.dynamics.sampling = Some(sim::SamplingProgress { done: 1, total: 273, stage: "scans", last_energy: -1.0, cancelled: false, skipped: 0, failed: 0, finished: false, target: None });
        st.absorb(&s);
        st.new_worker();
        assert!(matches!(st.timeline.last(), Some(Mark::Pass(_, Some(_), true))), "{:?}", st.timeline);
        // the next worker's model holds 7 quarantined samples: not a change
        let loaded = snapshot(&scene, Backend::Estimated, Some(ModelStatus { quarantined: 7, ..model() }));
        st.absorb(&loaded);
        assert!(!st.timeline.iter().any(|m| matches!(m, Mark::Quarantine(..))), "{:?}", st.timeline);
        // the same fit twice (a worker that loaded the model)
        let fit = crate::sim::FitPoint { epoch: 3, samples: 271, rmse_e: 2.1e-6, rmse_f: 1.6e-5, trusted: true, at: Instant::now() };
        let mut with_fit = snapshot(&scene, Backend::Estimated, Some(model()));
        with_fit.activity.fit_points = vec![fit];
        st.absorb(&with_fit);
        st.new_worker();
        st.absorb(&with_fit);
        assert_eq!(st.fit_points.len(), 1);
        assert_eq!(st.timeline.iter().filter(|m| matches!(m, Mark::Refit(..))).count(), 1);
    }

    /// M11: an old run's event arrives with the new scene: it is not matched against its geometry.
    #[test]
    fn an_old_runs_event_is_not_placed_by_the_new_scene() {
        let mut scene = Scene::preset("h2").unwrap();
        scene.nuclei[0].pos = [0.0, 0.0, 5.0]; // 1 Bohr from a face of the 49³ × 0.25 cell
        let mut st = activity::ActivityState::new();
        let with_atom = |e: sim::TeachEvent| sim::TeachEvent { atom: Some(0), box_affected: true, ..e };
        let old = with_atom(sim::TeachEvent::new(TeachOrigin::Fallback, TeachRule::Spread, TeachOutcome::Kept));
        std::thread::sleep(Duration::from_millis(3));
        let run = Instant::now();
        std::thread::sleep(Duration::from_millis(3));
        let new = with_atom(sim::TeachEvent::new(TeachOrigin::Fallback, TeachRule::Spread, TeachOutcome::Kept));
        let mut s = snapshot(&scene, Backend::Estimated, Some(model()));
        s.activity.events = vec![old, new];
        s.activity.run_started = Some(run);
        st.absorb(&s);
        assert!(!st.rows[0].near_face && st.rows[1].near_face);
    }
}

// ---- M7 Task 9: the ionic policy, the partial-core display and the words about pseudopotentials ----

/// The policy action replaces the scene (`Command::SetIonicPolicy`) only when it differs from the scene's, remembers it in
/// the panel, and presets loaded next are loaded under it (a preset file itself is all-electron); the words round-trip.
#[test]
fn ionic_action_round_trips() {
    for p in sim::IonicPolicy::ALL {
        let a = Action::Ionic(p);
        let text = a.describe().expect("a script form");
        assert_eq!(text, format!("ionic {}", p.word()));
        let words: Vec<&str> = text.split(' ').collect();
        assert_eq!(Action::parse(&words).unwrap(), a, "{text}");
    }
    for on in [true, false] {
        let a = Action::ShowPartialCore(on);
        let text = a.describe().unwrap();
        assert_eq!(text, format!("partial-core {on}"));
        let words: Vec<&str> = text.split(' ').collect();
        assert_eq!(Action::parse(&words).unwrap(), a);
    }
    assert!(Action::parse(&["ionic", "everything"]).unwrap_err().contains("none, cores, accepted or all"));

    let scene = Scene::preset("h2o").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let mut st = UiState::new("h2o", &scene);
    assert_eq!(st.ionic, DEFAULT_IONIC, "the app's default policy");
    let c = act(&mut st, &inp, Action::Ionic(sim::IonicPolicy::All));
    assert!(matches!(c.commands.as_slice(), [Command::SetIonicPolicy(sim::IonicPolicy::All)]) && st.ionic == sim::IonicPolicy::All, "{:?}", c.commands);
    let all = Scene { ionic: sim::IonicPolicy::All, ..scene.clone() };
    let inp_all = input(&all, None, &cam);
    assert!(act(&mut st, &inp_all, Action::Ionic(sim::IonicPolicy::All)).commands.is_empty(), "unchanged: nothing sent");
    let c = act(&mut st, &inp_all, Action::Preset("h2".into()));
    assert!(matches!(c.commands.first(), Some(Command::SetScene(s)) if s.ionic == sim::IonicPolicy::All), "{:?}", c.commands);
    let c = act(&mut st, &inp_all, Action::ShowPartialCore(true));
    assert!(matches!(c.commands.as_slice(), [Command::DisplayPartialCore(true)]) && st.show_partial_core);
    assert!(st.run_settings().iter().any(|c| matches!(c, DynCommand::SetDisplayPartialCore(true))), "a new worker is told");
    // a new worker is never told about a display that is off
    let st2 = UiState::new("h2", &scene);
    assert!(!st2.run_settings().iter().any(|c| matches!(c, DynCommand::SetDisplayPartialCore(_))));
}

/// M7 Task 11 fix round (controller ruling on the default): the app's default is `Accepted` — "accepted species" — and
/// Physics ▸ Ionic potential labels it "(default)"; while a pseudopotential is generated the status line says which and how
/// long it takes ("≈ 1 min" for the dimer-gated H/He), never a silent wait. Rendered: the frame's text shapes are searched.
#[test]
fn the_default_is_accepted_species_and_generation_is_said() {
    fn texts(ctx: &egui::Context, st: &mut UiState, inp: &PanelInput<'_>) -> String {
        fn walk(s: &egui::epaint::Shape, out: &mut String) {
            match s {
                egui::epaint::Shape::Text(t) => { out.push_str(t.galley.text()); out.push('\n'); }
                egui::epaint::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                _ => {}
            }
        }
        let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0))), ..Default::default() };
        let mut full = ctx.run_ui(raw, |root| { show(root, st, inp); });
        full.textures_delta.clear();
        let mut out = String::new();
        full.shapes.iter().for_each(|c| walk(&c.shape, &mut out));
        out
    }
    assert_eq!(DEFAULT_IONIC, sim::IonicPolicy::Accepted);
    assert_eq!((sim::IonicPolicy::Accepted.word(), sim::IonicPolicy::parse("accepted")), ("accepted", Some(sim::IonicPolicy::Accepted)));
    let scene = Scene { ionic: DEFAULT_IONIC, ..Scene::preset("h2").unwrap() };
    let cam = camera();
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    assert_eq!(st.ionic, sim::IonicPolicy::Accepted);
    let inp = input(&scene, None, &cam);
    egui::Popup::open_id(&ctx, menus::popup_id(menus::MenuId::Physics));
    let mut all = String::new();
    for _ in 0..3 { all = texts(&ctx, &mut st, &inp); }
    egui::Popup::close_all(&ctx);
    assert!(all.contains("accepted species (default)"), "the Physics menu:\n{all}");
    assert!(!all.contains("cores only (default)") && all.contains("cores only") && all.contains("all species"));
    // the status line while H is generated through its dimer gate, and nothing when nothing is
    let gen = PanelInput { pseudo_generating: Some((1, 12.4, true)), ..input(&scene, None, &cam) };
    let notes = overlays::chip_notes(&st, &gen);
    assert!(notes.iter().any(|(_, t)| t == "generating the H pseudopotential (≈ 1 min: its dimer check)… 12 s"), "{notes:?}");
    let o = PanelInput { pseudo_generating: Some((8, 3.0, false)), ..input(&scene, None, &cam) };
    assert!(overlays::chip_notes(&st, &o).iter().any(|(_, t)| t == "generating the O pseudopotential (seconds)… 3 s"));
    assert!(!overlays::chip_notes(&st, &inp).iter().any(|(_, t)| t.starts_with("generating")));
}

/// The policy edit and the display setting are classified like the neighbours: the first replaces the scene (an edit
/// that pauses and aborts), the second is display only.
#[test]
fn the_ionic_commands_are_classified() {
    let c = Command::SetIonicPolicy(sim::IonicPolicy::CoresOnly);
    assert!(c.is_edit() && c.pauses() && c.aborts());
    let mut s = Scene::preset("h2o").unwrap();
    c.apply(&mut s);
    assert_eq!(s.ionic, sim::IonicPolicy::CoresOnly, "the viewer's mirror follows at once");
    let d = Command::DisplayPartialCore(true);
    assert!(!d.is_edit() && !d.pauses() && !d.aborts());
}

/// M7 Task 11 carry (Task 10: CH₄ at Fast, h = 0.30, egg-box G 4.98e-4 Ha/Bohr, worse than the all-electron figure;
/// NLCC ruling 6: never refused, always said): with C's **real** generation record — certified for energies at its h_req,
/// for the egg-box force only to min(h_eb, h_req) — a C scene at Fast shows "egg-box force not certified at this grid"
/// in Physics ▸ Ionic potential, naming C and its certified grid; at Accurate (0.15) it does not. Rendered, not just
/// formatted: the frame's text shapes are searched.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only (generates C); cargo test -p app --release")]
fn carbon_at_fast_shows_the_egg_box_line() {
    fn texts(ctx: &egui::Context, st: &mut UiState, inp: &PanelInput<'_>) -> String {
        fn walk(s: &egui::epaint::Shape, out: &mut String) {
            match s {
                egui::epaint::Shape::Text(t) => { out.push_str(t.galley.text()); out.push('\n'); }
                egui::epaint::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                _ => {}
            }
        }
        let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0))), ..Default::default() };
        let mut full = ctx.run_ui(raw, |root| { show(root, st, inp); });
        full.textures_delta.clear();
        let mut out = String::new();
        full.shapes.iter().for_each(|c| walk(&c.shape, &mut out));
        out
    }
    // C's summary from the providers' own setup at Fast (a private cache: no other test thread generates into it)
    let at = |h: f64| {
        let a = 2.06 / 3f64.sqrt();
        let nuclei = [[0.0; 3], [a, a, a], [-a, -a, a], [-a, a, -a], [a, -a, -a]].iter().enumerate()
            .map(|(i, &pos)| crate::scene::SceneNucleus { z: if i == 0 { 6 } else { 1 }, a: None, pos, vel: [0.0; 3] }).collect();
        Scene { nuclei, charge: 0, multiplicity: None, grid_n: 2 * (6.0 / h).round() as usize + 1, grid_h: h, sigma: None, ionic: sim::IonicPolicy::CoresOnly }
    };
    let fast = at(crate::quality::parse("fast").expect("Fast has an h"));
    let private = std::sync::Mutex::new(providers::pseudo::PseudoCache::default());
    let setup = providers::pseudo::setup_in(&private, &fast.to_sim().unwrap(), fast.grid_h, &mut Vec::new()).expect("C is certified for energies at Fast").expect("C is pseudised");
    let pseudised = providers::pseudo::summaries(&setup);
    let c = pseudised.iter().find(|p| p.z == 6).expect("C's summary").clone();
    let grid = pseudo::egg_box_grid(&c).expect("C has a certified egg-box grid");
    eprintln!("C: h_req {:?}, h_eb {:?} → egg-box certified to {grid}; Fast h = {}", c.h_req, c.h_eb, fast.grid_h);
    assert!(grid < fast.grid_h, "C's egg-box force must not be certified at Fast for this check to mean anything");
    let cam = camera();
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    for (scene, shown) in [(fast.clone(), true), (at(0.15), false)] {
        let mut snap = snapshot(&scene, Backend::Gpu, None);
        snap.dynamics.eval.pseudised = pseudised.clone();
        let inp = input(&scene, Some(&snap), &cam);
        let mut st = UiState::new("h2o", &scene);
        egui::Popup::open_id(&ctx, menus::popup_id(menus::MenuId::Physics));
        let mut all = String::new();
        for _ in 0..3 { all = texts(&ctx, &mut st, &inp); }
        egui::Popup::close_all(&ctx);
        let line = format!("egg-box force not certified at this grid (h = {:.3} Bohr: C needs h ≤ {grid:.3})", scene.grid_h);
        assert_eq!(all.contains(&line), shown, "h = {}: {line:?} {} in the Physics menu:\n{all}", scene.grid_h, if shown { "missing" } else { "shown" });
    }
}

/// The words of the chrome: the egg-box line names who is not certified, the Quality marks come from the same
/// record, and a scene refused for its grid still gets them from the cache (never generating).
#[test]
fn the_quality_menu_marks_levels_a_pseudised_species_forbids() {
    use sim::PseudoSummary;
    let o = PseudoSummary { z: 8, z_v: 6.0, f: 1.0, local: "s-local".into(), r_c_max: 1.35, h_req: Some(0.20), h_eb: Some(0.15), r_core: 0.8, record: String::new() };
    assert_eq!(pseudo::needs_finer(&[o.clone()], 0.25).as_deref(), Some("needs h ≤ 0.20 (O)"));
    assert_eq!(pseudo::needs_finer(&[o.clone()], 0.30).as_deref(), Some("needs h ≤ 0.20 (O)"));
    assert_eq!(pseudo::needs_finer(&[o.clone()], 0.15), None);
    assert_eq!(pseudo::needs_finer(&[o], 0.20), None, "h = h_req is certified");
    assert_eq!(pseudo::needs_finer(&[], 0.30), None);
    // nothing pseudised and nothing generated: no marks, and no generation from the chrome
    let scene = Scene { ionic: sim::IonicPolicy::CoresOnly, ..Scene::preset("h2o").unwrap() };
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let _ = pseudo::pseudised(&inp);
    // the lookup the chrome uses, on a private cache (no other test thread can generate into it): nothing, and nothing generated
    let private = std::sync::Mutex::new(providers::pseudo::PseudoCache::default());
    assert!(providers::pseudo::cached_in(&private, &scene.to_sim().unwrap()).is_empty());
    assert_eq!(private.lock().unwrap().generated(), 0, "the chrome never generates");
}

/// User request 2026-10-18: the readouts show T averaged over the last wall-clock second, one
/// sample per step; a new nucleus count or a step going backwards starts over.
#[test]
fn temperature_readout_is_a_one_second_average() {
    let mut snap = empty_snapshot(Backend::Gpu, None);
    snap.dynamics.scene.nuclei = vec![sim::Nucleus::new(1, [0.0; 3]), sim::Nucleus::new(1, [1.4, 0.0, 0.0])];
    let mut avg = super::TempAverage::default();
    let mut at = |avg: &mut super::TempAverage, t: f64, step: u64, temp: f64| {
        snap.dynamics.step = step;
        snap.dynamics.temperature = temp;
        avg.observe(t, &snap.dynamics);
    };
    at(&mut avg, 0.0, 1, 100.0);
    at(&mut avg, 0.3, 2, 300.0);
    at(&mut avg, 0.3, 2, 900.0); // same step: not a new sample
    assert_eq!(avg.mean(), Some(200.0));
    at(&mut avg, 1.2, 3, 500.0); // the t = 0 sample leaves the window
    assert_eq!(avg.mean(), Some(400.0));
    at(&mut avg, 1.3, 1, 50.0); // the step went backwards: a reset
    assert_eq!(avg.mean(), Some(50.0));
    at(&mut avg, 5.0, 1, 50.0); // paused: the newest sample is kept
    assert_eq!(avg.mean(), Some(50.0));
}

/// User request 2026-10-18: Model ▸ Audits ▸ "audits on" switches audits off (cadence 0) and back
/// on at the last cadence; the off state shows the unsafe note.
#[test]
fn audits_switch_off_and_back_on_at_the_last_cadence() {
    let mut st = UiState::new("h2", &Scene::preset("h2").unwrap());
    st.audit_every = 500;
    st.audit_every_on = 500;
    assert!(super::dock::AUDIT_WHY.contains("spot checks") && super::dock::AUDIT_OFF_NOTE.contains("blind"));
    // the switch's two effects, as the dock applies them
    st.audit_every_on = st.audit_every.max(1);
    assert_eq!(st.audit_every_on, 500);
    st.audit_every = 0;
    assert_eq!(st.audit_every_on.max(1), 500, "turning back on restores the cadence");
}

/// User request 2026-10-18: Model ▸ Audits ▸ auto — the cadence is the steps of 10 s of wall time at
/// the measured speed, two significant figures, clamped; nothing while the speed is unmeasured.
#[test]
fn auto_audit_cadence_follows_the_play_speed() {
    assert_eq!(super::auto_audit_every(None, 0.25), None);
    assert_eq!(super::auto_audit_every(Some(0.0), 0.25), None);
    assert_eq!(super::auto_audit_every(Some(15.0), 0.25), Some(super::AUTO_AUDIT_MIN), "15 fs/s at 0.25 fs: 600 steps in 10 s, under the floor");
    assert_eq!(super::auto_audit_every(Some(157.0), 0.25), Some(6300));
    assert_eq!(super::auto_audit_every(Some(4000.0), 0.25), Some(160_000));
    assert_eq!(super::auto_audit_every(Some(1.0), 0.25), Some(super::AUTO_AUDIT_MIN), "slow play: the floor");
    assert_eq!(super::auto_audit_every(Some(1e9), 0.25), Some(super::AUTO_AUDIT_MAX), "the ceiling");
}

/// User request 2026-10-19: two docks. Activity, Readouts and Graph live on the left, Model, Orbitals,
/// Selection and Log on the right; each side has its own open flag and selected tab, a strip button
/// opens (or collapses) its own side only, and the search opens a control's own dock.
#[test]
fn the_two_docks_hold_their_own_tabs() {
    use super::DockSide;
    assert_eq!(DockTab::LEFT, [DockTab::Activity, DockTab::Readouts, DockTab::Graph], "the left dock's order");
    assert_eq!(DockTab::RIGHT, [DockTab::Model, DockTab::Orbitals, DockTab::Selection, DockTab::Log], "the right dock's order");
    assert!(DockTab::LEFT.iter().all(|t| t.side() == DockSide::Left) && DockTab::RIGHT.iter().all(|t| t.side() == DockSide::Right));
    assert_eq!(DockTab::ALL.len(), DockTab::LEFT.len() + DockTab::RIGHT.len());
    assert!(DockTab::ALL.iter().all(|t| t.side().tabs().contains(t)));

    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    // the defaults (user request 2026-10-19): the left dock open on Activity, the right on Model
    assert!(st.left_open && st.left_tab == DockTab::Activity && st.dock_open && st.dock_tab == DockTab::Model);
    // the rest starts from the left on Readouts and the right folded to its strip
    st.left_tab = DockTab::Readouts;
    st.dock_open = false;
    // a strip button opens its tab on its own side and leaves the other side alone
    click_widget(&ctx, &mut st, &inp, "dock-tab-Activity");
    assert!(st.tab_shown(DockTab::Activity) && !st.dock_open, "Activity opens the left dock");
    click_widget(&ctx, &mut st, &inp, "dock-tab-Selection");
    assert!(st.tab_shown(DockTab::Selection) && st.tab_shown(DockTab::Activity), "each side keeps its own tab");
    click_widget(&ctx, &mut st, &inp, "dock-tab-Graph");
    assert!(st.tab_shown(DockTab::Graph) && st.tab_shown(DockTab::Selection) && !st.tab_shown(DockTab::Activity));
    // the selected tab's button collapses its dock, only that one
    click_widget(&ctx, &mut st, &inp, "dock-tab-Selection");
    assert!(!st.dock_open && st.left_open && st.left_tab == DockTab::Graph);
    assert_eq!(st.dock_tab, DockTab::Selection, "the folded dock remembers its tab");
    // the chevrons: each toggles its own dock
    click_widget(&ctx, &mut st, &inp, "dock-chevron-Right");
    assert!(st.dock_open && st.left_open);
    click_widget(&ctx, &mut st, &inp, "dock-chevron-Left");
    assert!(st.dock_open && !st.left_open && st.left_tab == DockTab::Graph);
    click_widget(&ctx, &mut st, &inp, "dock-chevron-Left");
    assert!(st.left_open);
    // Ctrl K reveals a control in its own dock: the solve log on the left, the isotope on the right
    st.left_open = false;
    st.dock_open = false;
    let log = search::results("solve log", 1)[0];
    assert_eq!(log.place, search::Place::Dock(DockTab::Activity));
    search::pick(&ctx, &mut st, log);
    assert!(st.tab_shown(DockTab::Activity) && !st.dock_open);
    let guides = search::results("distance guides", 1)[0];
    search::pick(&ctx, &mut st, guides);
    assert!(st.tab_shown(DockTab::Selection) && st.tab_shown(DockTab::Activity));
    // the chrome renders every combination without a command or an error
    for (l, r) in [(false, false), (true, false), (false, true), (true, true)] {
        (st.left_open, st.dock_open) = (l, r);
        for w in [1400.0, 1000.0] {
            assert!(frame(&ctx, &mut st, &inp, w, vec![]).commands.is_empty(), "left {l} right {r} at {w}");
        }
    }
}

/// The docks' tab choice and open flags are display state: clicking them records no action.
#[test]
fn the_docks_state_is_not_recorded() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    let outs = [click_widget(&ctx, &mut st, &inp, "dock-tab-Graph"), click_widget(&ctx, &mut st, &inp, "dock-tab-Log"), click_widget(&ctx, &mut st, &inp, "dock-chevron-Left")];
    assert!(outs.iter().flatten().all(|o| o.actions.is_empty() && o.commands.is_empty()), "no action, no command");
}
