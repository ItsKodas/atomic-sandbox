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
        stale_bonds_line: None,
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
        body4_degree: None,
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
        forgotten: 0,
        suspects: Vec::new(),
        forgotten_samples: Vec::new(),
        superseded: 0,
        box_excluded: 0,
        unfitted: 0,
        box_affected: 0,
        trust_floor: Some(70),
        fit_vs_held: Some(crate::sim::fit_vs_held(271, 273, 0, 2, 0, 0, 0, 0, 0, 0)),
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
        class_loo: vec![sim::ClassLoo { class: "pair", samples: 273, force_rms: 0.009, force_max: 0.1, energy_rms: 0.05, energy_max: 0.152, trusted: true, refusal: None, calibration: 1.0, measured: None, measured_energy: None, worst_atom: None, evidence: 0, trusted_since: Some(1), inflation: 1.0, species: Vec::new() }],
        last_audit_this_fit: true,
        audits_box_unjudged: 0,
        solves: sim::SolveCounts::default(),
        trust_checks: Vec::new(),
        stuck_info: None,
        model_gen: 4,
        solve_s: Some(1.0),
        solve_s_measured: false,
        pass_eta_s: Some(25.0),
        pass_offer: None,
        no_model: None,
        retry_fit: false,
        source: sim::ModelSource::None,
        lone_note: None,
        collections: false,
        last_import: None,
        use_collection: false,
        source_is_autosave: false,
        specialised: Vec::new(),
        model_species: Vec::new(),
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
    // the old Rewind (v → −v) is Physics ▸ Reverse velocities since 2026-10-20
    assert!(matches!(one_dyn(&act(&mut st, &inp, Action::ReverseVelocities)), DynCommand::Rewind));
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
    assert_eq!(search::results("", 1000).len(), search::ENTRIES.len() + usize::from(crate::segment::SEGMENT_TRAINING));
    // Train this segment is deferred: Ctrl K does not offer it while the const is off
    assert_eq!(
        search::results("train this segment", 1000).iter().any(|e| std::ptr::eq(*e, &search::SEGMENT_ENTRY)),
        crate::segment::SEGMENT_TRAINING
    );
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
    assert_eq!((r[0].name, r[0].anchor, r[0].place), ("Trails", "m-display", search::Place::Dock(DockTab::View)));
    assert!(search::results("decay", 50).iter().any(|e| e.name == "Trails"));
    let inp = input(&scene, None, &cam);
    st.show_trails = true;
    // the View panel with every section open (its Display aids hold the trails)
    view_panel(&mut st, &dock::view::Section::ALL);
    for w in [1400.0, 1000.0] {
        for _ in 0..2 {
            assert!(frame(&ctx, &mut st, &inp, w, vec![]).commands.is_empty());
        }
    }
    probed_rect("anchor-m-display"); // drawn
    assert!(st.show_trails && st.trail_fs == 100.0 && st.trail_opacity == 0.7, "drawing changes nothing");
}

/// Every menu popup (and the scene chip's) fits inside the default 1400 × 900 window, at the
/// start and with every conditional row shown (the estimated solver's draft line): the View menu
/// once measured 1011 pt tall (2026-10-20), its last rows out of reach. Past the space below the
/// top bar a menu scrolls (`menus::menu_popup`). The View menu's controls are the View panel's
/// since docking Task 4: with every section open and every conditional row shown (dots look,
/// Camera, trails) it scrolls, and Ctrl K still reveals its groups inside it.
#[test]
fn every_menu_popup_fits_the_default_window() {
    let scene = Scene::preset("h2o").unwrap();
    let cam = camera();
    let snap = snapshot(&scene, Backend::Estimated, Some(model()));
    let inp = input(&scene, Some(&snap), &cam);
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0));
    let ids: Vec<(&str, egui::Id)> =
        menus::MenuId::ALL.iter().map(|&m| (m.title(), menus::popup_id(m))).chain([("scene chip", menus::scene_chip_id())]).collect();
    for worst in [false, true] {
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2o", &scene);
        st.gpu_available = true;
        if worst {
            st.backend = Backend::Estimated;
            st.draft = true;
            st.volume_mode = VOLUME_DOTS;
            st.show = ShowMode::Coloured;
            st.camera_view = true;
            st.show_trails = true;
        }
        for &(name, id) in &ids {
            egui::Popup::open_id(&ctx, id);
            for _ in 0..3 {
                frame(&ctx, &mut st, &inp, 1400.0, vec![]);
            }
            let r = ctx.memory(|m| m.area_rect(id)).unwrap_or_else(|| panic!("{name}: no popup area"));
            egui::Popup::close_all(&ctx);
            frame(&ctx, &mut st, &inp, 1400.0, vec![]);
            assert!(screen.contains_rect(r), "{name} (worst {worst}): {r:?} is not inside {screen:?}");
            if !worst {
                // at the start nothing needs the scroll: the capped View menu would end at 900
                assert!(r.max.y < 899.0, "{name} scrolls at the start: {r:?}");
                eprintln!("{name}: {:.0} pt tall", r.height());
            }
        }
        if worst {
            // with every section open and every row shown the View panel scrolls, and Ctrl K still
            // reveals its groups: Bonds & labels, then the last, Display aids
            st.view_sections = dock::view::Sections::all(true);
            for (query, anchor, probe) in [("bond lines", "m-bonds", "show-bonds"), ("training is solving", "m-display", "show-ghost")] {
                let e = search::results(query, 1)[0];
                assert_eq!(e.anchor, anchor);
                search::pick(&ctx, &mut st, e);
                // the scroll is animated: these frames carry a clock (`frame`'s do not)
                for i in 0..20 {
                    let raw = egui::RawInput { screen_rect: Some(screen), time: Some(100.0 + 0.05 * i as f64), ..Default::default() };
                    let mut full = ctx.run_ui(raw, |root| {
                        show(root, &mut st, &inp);
                    });
                    full.textures_delta.clear();
                }
                let (b, panel) = (probed_rect(probe), probed_rect("view-panel"));
                assert!(screen.contains_rect(b) && panel.contains_rect(b), "the revealed {probe} {b:?} is scrolled into the View panel's view {panel:?}");
            }
        }
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
    frame_sized(ctx, st, inp, egui::vec2(w, 900.0), events)
}

fn frame_sized(ctx: &egui::Context, st: &mut UiState, inp: &PanelInput<'_>, size: egui::Vec2, events: Vec<egui::Event>) -> PanelOutput {
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
        events,
        ..Default::default()
    };
    let mut out = None;
    let mut full = ctx.run_ui(raw, |root| out = Some(show(root, st, inp)));
    // the window hands these to the painter; dropping them unapplied trips epaint's debug assertion
    full.textures_delta.clear();
    out.unwrap()
}

/// Every layout (wide and compact; each panel shown and every panel closed; every menu open; the
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
    rich.activity.knowledge = Some(std::sync::Arc::new(knowledge::fixture_map()));
    st.activity.absorb(&rich);
    st.show_knowledge = true; // the overlay's legend over every layout
    st.view_sections = super::dock::view::Sections::all(true); // every View-panel anchor drawn
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
            for tab in DockTab::PANELS {
                st.open_tab(tab);
                for _ in 0..2 {
                    let out = frame(&ctx, &mut st, &inp, w, vec![]);
                    assert!(out.commands.is_empty() && !out.toggle_play && !out.deselect && out.error.is_none(), "{tab:?} at {w}");
                }
            }
            for tab in DockTab::PANELS {
                st.workspace.close_panel(tab);
            }
            frame(&ctx, &mut st, &inp, w, vec![]);
            st.reset_layout();
            for id in menus::MenuId::ALL.map(menus::popup_id).into_iter().chain([menus::scene_chip_id()]) {
                egui::Popup::open_id(&ctx, id);
                for _ in 0..2 {
                    assert!(frame(&ctx, &mut st, &inp, w, vec![]).commands.is_empty());
                }
            }
            egui::Popup::close_all(&ctx);
            // the capture settings popover (its pop-out button)
            st.capture.open = true;
            assert!(frame(&ctx, &mut st, &inp, w, vec![]).commands.is_empty());
            st.capture.open = false;
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

/// Ctrl K opens the search, Enter reveals the best match (its panel is shown), Esc closes the
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
    st.open_tab(DockTab::Readouts); // Selection's leaf shows another tab
    assert!(!st.tab_shown(DockTab::Selection));
    frame(&ctx, &mut st, &inp, 1400.0, vec![key(egui::Key::Enter, egui::Modifiers::NONE)]);
    assert!(!st.search.open);
    assert!(st.tab_shown(DockTab::Selection) && !st.tab_shown(DockTab::Readouts), "the control's panel is shown in its leaf");
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
    press_widget_w(ctx, st, inp, 1400.0, name, drag)
}

/// [`press_widget`] in a window `w` points wide (900 high; below 1240 the compact layout).
fn press_widget_w(ctx: &egui::Context, st: &mut UiState, inp: &PanelInput<'_>, w: f32, name: &str, drag: f32) -> Vec<PanelOutput> {
    for _ in 0..3 {
        frame(ctx, st, inp, w, vec![]);
    }
    let c = probed_rect(name).center();
    let mut outs = vec![frame(ctx, st, inp, w, vec![egui::Event::PointerMoved(c), pointer(c, true)])];
    let mut at = c;
    for _ in 0..4 {
        if drag != 0.0 {
            at.x += drag / 4.0;
            outs.push(frame(ctx, st, inp, w, vec![egui::Event::PointerMoved(at)]));
        }
    }
    outs.push(frame(ctx, st, inp, w, vec![pointer(at, false)]));
    outs.push(frame(ctx, st, inp, w, vec![egui::Event::PointerGone]));
    outs
}

/// The old right dock's geometry, for a panel's click and on-screen-text tests written against it:
/// `tab` alone in one leaf right of the 3D view — 336 pt wide at 1400 (24 %), the full height —
/// where the default layout (docking spec D-9) gives it a shorter leaf that would clip the widget
/// or text the test looks for. Another panel shown after it joins the same leaf.
fn alone(st: &mut UiState, tab: DockTab) {
    st.workspace.set_node(super::workspace::layout::parse(&format!("H(0.76,3D,[{}*])", tab.name())).unwrap());
}

/// The View panel (docking Task 4: the View menu's controls) as [`alone`] gives it, with only the
/// sections `open` unfolded, so the control a test clicks is on screen.
fn view_panel(st: &mut UiState, open: &[dock::view::Section]) {
    alone(st, DockTab::View);
    st.view_sections = dock::view::Sections::all(false);
    for &s in open {
        st.view_sections.set(s, true);
    }
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
    // the tab is taller than the 900 pt frame (the Forget button sits under the toolbar's edge):
    // scroll each group into view the way Ctrl K does, letting the scroll animation settle
    let reveal = |st: &mut UiState, anchor: &str| {
        search::pick(&ctx, st, search::ENTRIES.iter().find(|e| e.anchor == anchor).unwrap());
        for _ in 0..40 {
            frame(&ctx, st, &inp, 1400.0, vec![]);
        }
    };
    reveal(&mut st, "d-model-file");
    assert!(dyn_commands(&click_widget(&ctx, &mut st, &inp, "forget")).is_empty(), "the first click arms");
    let c = dyn_commands(&click_widget(&ctx, &mut st, &inp, "forget"));
    assert!(matches!(c.as_slice(), [DynCommand::ForgetModel]), "{c:?}");
    reveal(&mut st, "d-sample");
    let seed = st.plan.seed;
    assert!(dyn_commands(&click_widget(&ctx, &mut st, &inp, "sample")).is_empty(), "a long pass arms first");
    let c = dyn_commands(&click_widget(&ctx, &mut st, &inp, "sample"));
    assert!(matches!(c.as_slice(), [DynCommand::StartSampling(p)] if p.seed == seed), "{c:?}");
    assert_eq!(st.plan.seed, seed + 1);
}

/// The 2026-10-20 PH₃ data loss: samples held but no model formed — the Model tab shows the reason
/// and its Retry sends `RetryFit`, and so does the Activity tab's trust group.
#[test]
fn a_fit_that_did_not_form_shows_why_and_retries() {
    let scene = Scene::preset("ph3").unwrap();
    let cam = camera();
    let why = "no model yet: the P atom reference did not solve (2 cold attempts: SCF stalled)";
    let m = ModelStatus { fitted: false, trusted: false, fit: None, samples: 41, no_model: Some(why.into()), retry_fit: true, ..model() };
    let snap = snapshot(&scene, Backend::Estimated, Some(m.clone()));
    let inp = input(&scene, Some(&snap), &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("ph3", &scene);
    st.backend = Backend::Estimated;
    alone(&mut st, DockTab::Model);
    for _ in 0..3 {
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
    }
    let c = dyn_commands(&click_widget(&ctx, &mut st, &inp, "retry-fit"));
    assert!(matches!(c.as_slice(), [DynCommand::RetryFit]), "{c:?}");
    // the Activity tab's trust group says the same and retries too
    st.open_tab(DockTab::Activity);
    for _ in 0..3 {
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
    }
    let c = dyn_commands(&click_widget(&ctx, &mut st, &inp, "retry-fit"));
    assert!(matches!(c.as_slice(), [DynCommand::RetryFit]), "{c:?}");
}

/// Spec 2026-11-12 R2: the Model tab's "Worst-fitting samples" — the rows (the spec's text), the
/// caption, the forgotten list; clicking Forget and Restore emits `Action::ForgetSample` /
/// `RestoreSample`, which send `Command::ForgetSample` / `RestoreSample` and are recorded as not
/// replayable; with no model, or no ranking yet, the section says why. Rendered: the frame's text
/// shapes are searched.
#[test]
fn the_worst_fitting_samples_forget_and_restore_one() {
    fn texts(ctx: &egui::Context, st: &mut UiState, inp: &PanelInput<'_>) -> String {
        fn walk(s: &egui::epaint::Shape, out: &mut String) {
            match s {
                egui::epaint::Shape::Text(t) => {
                    out.push_str(t.galley.text());
                    out.push('\n');
                }
                egui::epaint::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                _ => {}
            }
        }
        let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0))), ..Default::default() };
        let mut full = ctx.run_ui(raw, |root| {
            show(root, st, inp);
        });
        full.textures_delta.clear();
        let mut out = String::new();
        full.shapes.iter().for_each(|c| walk(&c.shape, &mut out));
        out
    }
    // the id carries a check value (fix round 1, M8): the buttons send it as given
    let suspect = |index, formula: &str, origin: &str, loo_ratio| sim::SampleSuspect { id: sim::SampleId { index, key: 0xabc0 + index as u64 }, formula: formula.into(), origin: origin.into(), loo_ratio };
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let m = ModelStatus {
        suspects: vec![suspect(1804, "HO", "fallback", Some(38.46)), suspect(1133, "H2O", "pass perturbation", Some(5.23))],
        forgotten_samples: vec![suspect(40, "H2", "audit", None)],
        forgotten: 1,
        ..model()
    };
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    st.backend = Backend::Estimated;
    alone(&mut st, DockTab::Model);
    // the section is collapsed by default: open it the way Ctrl K does
    let reveal = |st: &mut UiState, inp: &PanelInput<'_>| {
        search::pick(&ctx, st, search::ENTRIES.iter().find(|e| e.anchor == "d-suspects").unwrap());
        for _ in 0..40 {
            frame(&ctx, st, inp, 1400.0, vec![]);
        }
    };
    let snap = snapshot(&scene, Backend::Estimated, Some(m.clone()));
    let inp = input(&scene, Some(&snap), &cam);
    reveal(&mut st, &inp);
    let t = texts(&ctx, &mut st, &inp);
    assert!(t.contains("#1804 · HO · fallback · 38× its bound"), "{t}");
    assert!(t.contains("#1133 · H2O · pass perturbation · 5.2× its bound"), "{t}");
    assert!(t.contains(super::dock::model::SUSPECTS_CAPTION), "{t}");
    assert!(t.contains("forgotten (1): held out of the fit until restored") && t.contains("#40 · H2 · audit · not in the fit"), "{t}");
    let sent = |outs: &[PanelOutput]| outs.iter().flat_map(|o| o.commands.clone()).collect::<Vec<_>>();
    let c = sent(&click_widget(&ctx, &mut st, &inp, "forget-sample-1804"));
    assert!(matches!(c.as_slice(), [Command::ForgetSample(sim::SampleId { index: 1804, key })] if *key == 0xabc0 + 1804), "{c:?}");
    let c = sent(&click_widget(&ctx, &mut st, &inp, "restore-sample-40"));
    assert!(matches!(c.as_slice(), [Command::RestoreSample(sim::SampleId { index: 40, key })] if *key == 0xabc0 + 40), "{c:?}");
    let id = sim::SampleId { index: 1804, key: 1 };
    assert_eq!(crate::replay::unreplayable_kind(&Action::ForgetSample(id)), Some("forget-sample"));
    assert_eq!(crate::replay::unreplayable_kind(&Action::RestoreSample(id)), Some("restore-sample"));
    // the forgotten count beside the held one
    assert_eq!(crate::sim::fit_vs_held(270, 273, 0, 2, 0, 1, 0, 0, 0, 0), "270 of 273 held: 2 quarantined, 1 forgotten");
    // no ranking yet (a model, no list), and no model: the section says why
    for (m, why) in [
        (ModelStatus { suspects: Vec::new(), forgotten_samples: Vec::new(), ..model() }, super::dock::model::SUSPECTS_NO_RANKING),
        (ModelStatus { fitted: false, trusted: false, fit: None, suspects: Vec::new(), forgotten_samples: Vec::new(), ..model() }, super::dock::model::SUSPECTS_NO_MODEL),
    ] {
        let snap = snapshot(&scene, Backend::Estimated, Some(m));
        let inp = input(&scene, Some(&snap), &cam);
        reveal(&mut st, &inp);
        let t = texts(&ctx, &mut st, &inp);
        assert!(t.contains(why), "{why}: {t}");
        assert!(!t.contains("#1804"), "{t}");
    }
}

/// M8a Task 4b: the Model tab's source line — shared (writable, read-only), a file, per-scene — and
/// the import's outcome line; the tab draws them (and the switch) without sending anything; the
/// switch's and the import's actions send their commands and are recorded as not replayable.
#[test]
fn the_model_tab_says_where_the_model_learns() {
    use crate::sim::source_line;
    let coll = |read_only| sim::ModelSource::Collection { slug: "h0.15-auto-accepted-gpu-1e-6".into(), samples: 412, read_only };
    let m = ModelStatus { source: coll(false), collections: true, ..model() };
    assert_eq!(source_line(&m), "model: shared — h0.15-auto-accepted-gpu-1e-6 · 412 samples");
    assert_eq!(source_line(&ModelStatus { source: coll(true), ..m.clone() }), "model: shared — h0.15-auto-accepted-gpu-1e-6 · 412 samples · read-only — open in another window");
    assert_eq!(source_line(&ModelStatus { source: sim::ModelSource::File("C:/m/h2o.model".into()), ..m.clone() }), "file: h2o.model");
    assert_eq!(source_line(&ModelStatus { source: sim::ModelSource::None, ..m.clone() }), "model: this scene only (not shared)");
    // fix round 1 (review Minor 2): this session's autosave of a picked file is named by the pick
    let auto = ModelStatus { source: sim::ModelSource::File("C:/t/atomic-sandbox-42-1.model".into()), source_is_autosave: true, ..m.clone() };
    assert_eq!(crate::sim::source_line_with(&auto, Some(std::path::Path::new("C:/m/water.model"))), "file: water.model");
    assert_eq!(source_line(&auto), "file: the model file picked earlier in this session");
    let r = sim::ImportSummary { imported: 30, refused: vec![(Some(4), "box status unknown".into()), (None, "another solver surface".into())], notes: vec![] };
    assert_eq!(super::dock::model::import_line(&r), "last import: imported 30, refused 2");
    // drawn (with an import's outcome) without sending anything
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let m = ModelStatus { last_import: Some(Ok(r)), ..m };
    let snap = snapshot(&scene, Backend::Estimated, Some(m));
    let inp = input(&scene, Some(&snap), &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    st.backend = Backend::Estimated;
    st.open_tab(DockTab::Model);
    for _ in 0..3 {
        assert!(frame(&ctx, &mut st, &inp, 1400.0, vec![]).commands.is_empty());
    }
    // the actions
    let mut out = PanelOutput::default();
    apply(&mut st, &inp, Action::UseCollection, &mut out);
    assert!(matches!(out.commands.as_slice(), [Command::UseModel(crate::sim::ModelChoice::Collection)]), "{:?}", out.commands);
    let mut out = PanelOutput::default();
    apply(&mut st, &inp, Action::ImportModel("x.model".into()), &mut out);
    assert!(matches!(out.commands.as_slice(), [Command::ImportModel(p)] if p.ends_with("x.model")), "{:?}", out.commands);
    assert_eq!(crate::replay::unreplayable_kind(&Action::UseCollection), Some("use-collection"));
    assert_eq!(crate::replay::unreplayable_kind(&Action::ImportModel("x.model".into())), Some("import-model"));
    // M8a Task 7: "Save a specialised model…" — the scene's species preselected by the drawn tab,
    // the action sends the picker's species, nothing ticked is an error; a loaded one says so
    assert_eq!(st.specialise.0, vec![1], "H₂'s species preselected");
    let mut out = PanelOutput::default();
    apply(&mut st, &inp, Action::SaveSpecialised("hc.model".into()), &mut out);
    assert!(matches!(out.commands.as_slice(), [Command::SaveSpecialised { path, species }] if path.ends_with("hc.model") && species == &vec![1]), "{:?}", out.commands);
    st.specialise.0.clear();
    let mut out = PanelOutput::default();
    apply(&mut st, &inp, Action::SaveSpecialised("hc.model".into()), &mut out);
    assert!(out.commands.is_empty() && out.error.is_some());
    assert_eq!(crate::replay::unreplayable_kind(&Action::SaveSpecialised("x.model".into())), Some("specialised"));
    let spec = ModelStatus { source: sim::ModelSource::File("C:/m/hc.model".into()), specialised: vec![1, 6], ..model() };
    assert_eq!(source_line(&spec), "file: hc.model (specialised)");
}

/// The M5b controls in their new homes, clicked: Scene ▸ Container's walls checkbox, the View
/// panel's cloud refresh, Physics ▸ draft mode (estimated solver).
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
    // the Model panel shows at start (the default layout's first leaf), whose draft checkbox is
    // probed as "draft" too and drawn after the menus: show Knowledge in its leaf instead so the
    // probe is the menu's
    st.open_tab(DockTab::Knowledge);
    assert!(!st.tab_shown(DockTab::Model));
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
    // the View panel's What you see (the panel alone: the Model panel, and its "draft", go)
    view_panel(&mut st, &[dock::view::Section::What]);
    let outs = click_widget(&ctx, &mut st, &inp, "cloud-refresh");
    let clouds: Vec<_> = outs.iter().flat_map(|o| &o.commands).filter(|c| matches!(c, Command::SetCloud { superposition: true, refresh_s: Some(x) } if *x == 2.0)).collect();
    assert_eq!(clouds.len(), 1);
    let outs = in_menu(&mut st, menus::MenuId::Physics, "draft");
    let drafts: Vec<_> = outs.iter().flat_map(|o| &o.commands).filter(|c| matches!(c, Command::SetDraft(true))).collect();
    assert_eq!(drafts.len(), 1);
    assert!(st.draft);
}

// ---- ⚠ Model only (user request 2026-11-12) ----

/// The commands a run of frames sent for draft mode and the audit cadence, in order.
fn draft_and_audit(outs: &[PanelOutput]) -> Vec<String> {
    outs.iter()
        .flat_map(|o| &o.commands)
        .filter(|c| matches!(c, Command::SetDraft(_) | Command::SetAudit(_)))
        .map(|c| format!("{c:?}"))
        .collect()
}

/// `Action::ModelOnly`: on = draft on and audits 0 (the previous two kept), off = both restored
/// exactly; refused off Estimated; leaving Estimated restores; a hand change of draft or the
/// cadence turns it off and restores nothing over it (the other setting stays).
#[test]
fn model_only_saves_and_restores_draft_and_audits() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let fresh = || {
        let mut st = UiState::new("h2", &scene);
        st.gpu_available = true;
        st.backend = Backend::Estimated;
        st
    };
    let mut st = fresh();
    assert!(st.model_only.is_none() && overlays::model_only_chip(&st).is_none(), "every session starts with it off");
    act(&mut st, &inp, Action::SetAudit(500));
    let out = act(&mut st, &inp, Action::ModelOnly(true));
    assert!(matches!(out.commands.as_slice(), [Command::SetDraft(true), Command::SetAudit(0)]), "{:?}", out.commands);
    assert!(st.draft && st.audit_every == 0);
    assert_eq!(st.model_only, Some(ModelOnlySaved { draft: false, audit_every: 500 }));
    assert_eq!(st.audit_every_on, 500, "\"audits on\" would bring back the replaced cadence");
    assert_eq!(overlays::model_only_chip(&st), Some("model only — no DFT checks"));
    assert!(act(&mut st, &inp, Action::ModelOnly(true)).commands.is_empty(), "already on: nothing");
    let out = act(&mut st, &inp, Action::ModelOnly(false));
    assert!(matches!(out.commands.as_slice(), [Command::SetDraft(false), Command::SetAudit(500)]), "{:?}", out.commands);
    assert!(!st.draft && st.audit_every == 500 && st.model_only.is_none() && overlays::model_only_chip(&st).is_none());
    assert!(act(&mut st, &inp, Action::ModelOnly(false)).commands.is_empty(), "already off: nothing");
    // draft already on before: it stays on after
    act(&mut st, &inp, Action::Draft(true));
    act(&mut st, &inp, Action::ModelOnly(true));
    let out = act(&mut st, &inp, Action::ModelOnly(false));
    assert!(matches!(out.commands.as_slice(), [Command::SetDraft(true), Command::SetAudit(500)]), "{:?}", out.commands);
    // the solver leaves Estimated: off, restored
    let mut st = fresh();
    act(&mut st, &inp, Action::SetAudit(700));
    act(&mut st, &inp, Action::ModelOnly(true));
    let out = act(&mut st, &inp, Action::Backend(Backend::Gpu));
    assert!(matches!(out.commands.as_slice(), [Command::SetDraft(false), Command::SetAudit(700)]), "{:?}", out.commands);
    assert!(st.backend == Backend::Gpu && st.model_only.is_none() && !st.draft && st.audit_every == 700);
    // not on another solver (the toggle is hidden there; a script could still ask)
    let out = act(&mut st, &inp, Action::ModelOnly(true));
    assert!(out.commands.is_empty() && out.error.is_some() && st.model_only.is_none());
    // a hand change of draft: off, the user's draft kept, the audits left at 0 (nothing restored)
    let mut st = fresh();
    act(&mut st, &inp, Action::ModelOnly(true));
    let out = act(&mut st, &inp, Action::Draft(false));
    assert!(matches!(out.commands.as_slice(), [Command::SetDraft(false)]), "{:?}", out.commands);
    assert!(st.model_only.is_none() && !st.draft && st.audit_every == 0);
    assert!(st.status.contains("by hand"), "{}", st.status);
    // a hand change of the cadence: off, the cadence kept, draft left on
    let mut st = fresh();
    act(&mut st, &inp, Action::ModelOnly(true));
    let out = act(&mut st, &inp, Action::SetAudit(300));
    assert!(matches!(out.commands.as_slice(), [Command::SetAudit(300)]), "{:?}", out.commands);
    assert!(st.model_only.is_none() && st.draft && st.audit_every == 300);
    // Ctrl K finds it by either name
    for q in ["model only", "no DFT checks"] {
        assert_eq!(search::results(q, 1)[0].anchor, "tb-model-only", "{q}");
    }
}

/// The toolbar's ⚠ Model only, clicked: absent unless Estimated is the solver (and in the compact
/// slot of a window too narrow for it, 1000 pt); on sends draft on
/// and audits 0 (recorded as one `ModelOnly(true)`) and puts the pill in the status chip; off
/// restores a non-default cadence; switching to GPU while on restores; the Model panel's "audits
/// on" (a hand change) turns it off and keeps the user's choice.
#[test]
fn clicking_model_only_toggles_restores_and_hides_off_estimated() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let gpu = snapshot(&scene, Backend::Gpu, None);
    let est = snapshot(&scene, Backend::Estimated, Some(model()));
    let mut st = UiState::new("h2", &scene);
    st.gpu_available = true;
    st.backend = Backend::Gpu;
    let shown = |st: &mut UiState, inp: &PanelInput<'_>, w: f32| {
        PROBES.with(|p| p.borrow_mut().clear());
        frame(&ctx, st, inp, w, vec![]);
        PROBES.with(|p| p.borrow().contains_key("model-only"))
    };
    for w in [1400.0, 1000.0] {
        assert!(!shown(&mut st, &input(&scene, Some(&gpu), &cam), w), "{w}: hidden under GPU");
    }
    st.backend = Backend::Estimated;
    let inp = input(&scene, Some(&est), &cam);
    for w in [1400.0, 1240.0] {
        assert!(shown(&mut st, &inp, w), "{w}: shown under Estimated");
    }
    // the compact slot at 1000 has no room for it: not drawn, Ctrl K's anchor is Estimated
    assert!(!shown(&mut st, &inp, 1000.0), "1000: no room in the compact slot");
    assert_eq!(probed_rect("anchor-tb-model-only"), probed_rect("solver-estimated"));
    frame(&ctx, &mut st, &inp, 1400.0, vec![]);
    assert!(!PROBES.with(|p| p.borrow().contains_key("model-only-chip")), "no chip pill while off");
    act(&mut st, &inp, Action::SetAudit(500));
    let outs = click_widget(&ctx, &mut st, &inp, "model-only");
    assert_eq!(draft_and_audit(&outs), ["SetDraft(true)", "SetAudit(0)"]);
    let recorded: Vec<_> = outs.iter().flat_map(|o| &o.actions).filter(|a| matches!(a, Action::ModelOnly(_))).collect();
    assert_eq!(recorded, [&Action::ModelOnly(true)], "recorded as one action");
    assert!(st.draft && st.audit_every == 0 && st.model_only.is_some());
    probed_rect("model-only-chip"); // the chip's pill is drawn
    let outs = click_widget(&ctx, &mut st, &inp, "model-only");
    assert_eq!(draft_and_audit(&outs), ["SetDraft(false)", "SetAudit(500)"]);
    assert!(!st.draft && st.audit_every == 500 && st.model_only.is_none());
    // on, then GPU: off and restored
    click_widget(&ctx, &mut st, &inp, "model-only");
    let outs = click_widget(&ctx, &mut st, &inp, "solver-gpu");
    assert_eq!(draft_and_audit(&outs), ["SetDraft(false)", "SetAudit(500)"]);
    assert!(st.backend == Backend::Gpu && st.model_only.is_none() && !st.draft && st.audit_every == 500);
    // on, then the Model panel's "audits on" by hand: off, the replaced cadence back, draft still on
    st.backend = Backend::Estimated;
    click_widget(&ctx, &mut st, &inp, "model-only");
    alone(&mut st, DockTab::Model);
    let outs = click_widget(&ctx, &mut st, &inp, "audits-on");
    assert_eq!(draft_and_audit(&outs), ["SetAudit(500)"]);
    assert!(st.model_only.is_none() && st.draft && st.audit_every == 500);
    // the status chip's pill turns it off (the way off at 1000 pt, where the toggle is not drawn)
    st.draft = false;
    click_widget(&ctx, &mut st, &inp, "model-only");
    assert!(st.model_only.is_some());
    PROBES.with(|p| p.borrow_mut().clear());
    let outs = press_widget_w(&ctx, &mut st, &inp, 1000.0, "model-only-chip", 0.0);
    assert!(!PROBES.with(|p| p.borrow().contains_key("model-only")), "1000: the toggle is not drawn");
    let recorded: Vec<_> = outs.iter().flat_map(|o| &o.actions).filter(|a| matches!(a, Action::ModelOnly(_))).collect();
    assert_eq!(recorded, [&Action::ModelOnly(false)], "the pill sends Model only off");
    assert_eq!(draft_and_audit(&outs), ["SetDraft(false)", "SetAudit(500)"]);
    assert!(st.model_only.is_none() && !st.draft && st.audit_every == 500);
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

/// Final review 4: while Train this segment is deferred, File ▸ Load (and a replay's `scene` line,
/// the same action) loads a scene file's `train_mask` line as nothing — the scene sent has no mask
/// — and the status line says so.
#[test]
fn a_scene_files_train_mask_is_ignored_while_segments_are_deferred() {
    let dir = std::env::temp_dir().join(format!("atomic-ui-mask-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("seg.scene").display().to_string();
    std::fs::write(&path, "grid 49 0.25
train_mask 1 0 1
nucleus 6 0 0 0
nucleus 1 2 0 0
nucleus 1 -2 0 0
").unwrap();
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let mut st = UiState::new("h2", &scene);
    st.scene_path = path.clone();
    let out = act(&mut st, &inp, Action::LoadScene { sidecar: false });
    assert!(out.error.is_none(), "{:?}", out.error);
    let sent = match out.commands.first() {
        Some(Command::SetScene(s)) => s.clone(),
        other => panic!("{other:?}"),
    };
    assert_eq!(sent.nuclei.len(), 3, "the scene loads");
    if crate::segment::SEGMENT_TRAINING {
        assert_eq!(sent.train_mask, Some(vec![true, false, true]));
    } else {
        assert_eq!(sent.train_mask, None, "the mask is not loaded");
        assert!(st.status.contains("train_mask ignored: Train this segment is deferred"), "{}", st.status);
    }
    let _ = std::fs::remove_dir_all(&dir);
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
        for tab in DockTab::PANELS {
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
    // the View panel: Bonds & labels, then Display aids
    view_panel(&mut st, &[dock::view::Section::Bonds]);
    assert!(click_widget(&ctx, &mut st, &inp, "show-bonds").iter().all(|o| o.commands.is_empty()));
    assert!(!st.show_bonds);
    // View ▸ Display aids ▸ show what training is solving: on by default, display only (no command,
    // no recorded action), and found by Ctrl K
    assert!(st.show_ghost);
    view_panel(&mut st, &[dock::view::Section::Aids]);
    let outs = click_widget(&ctx, &mut st, &inp, "show-ghost");
    assert!(outs.iter().all(|o| o.commands.is_empty() && o.actions.is_empty()), "display only");
    assert!(!st.show_ghost);
    click_widget(&ctx, &mut st, &inp, "show-ghost");
    assert!(st.show_ghost);
    assert_eq!(search::results("training is solving", 3)[0].name, "Show what training is solving");
    assert_eq!(search::results("ghost", 1)[0].name, "Show what training is solving");
    assert!(matches!(search::results("what training", 1)[0].place, search::Place::Dock(DockTab::View)));
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

/// An estimated snapshot of `scene` whose bond graph is an earlier DFT step's (`stale`) or this
/// one's.
fn bonds_snapshot(scene: &Scene, backend: Backend, stale: bool) -> Snapshot {
    let mut s = snapshot(scene, backend, (backend == Backend::Estimated).then(model));
    s.dynamics.bonds = Some(graph(scene.nuclei.len(), &[(0, 1)], vec![(0..scene.nuclei.len()).collect()], 12.5 / units::au_time_to_fs(1.0)));
    s.dynamics.bonds_stale = stale;
    s
}

/// User request 2026-11-13: View ▸ Bonds & labels ▸ guess bonds from experience is display state,
/// on by default, in the profile as `guess_bonds` (an old profile without it, or an unreadable
/// value, reads as on), put back by Reset view settings, and in Ctrl K.
#[test]
fn the_guess_from_experience_persists_in_the_profile() {
    let scene = Scene::preset("h2").unwrap();
    let mut st = UiState::new("h2", &scene);
    assert!(st.guess_bonds, "on by default");
    assert!(profile::capture(&st).contains("\nguess_bonds = true\n"));
    st.guess_bonds = false;
    let text = profile::capture(&st);
    assert!(text.contains("\nguess_bonds = false\n"), "{text}");
    let mut back = UiState::new("h2", &scene);
    profile::apply(&mut back, &text);
    assert!(!back.guess_bonds);
    let old: String = text.lines().filter(|l| !l.starts_with("guess_bonds")).map(|l| format!("{l}\n")).collect();
    profile::apply(&mut back, &old);
    assert!(back.guess_bonds, "an old profile: on");
    back.guess_bonds = false;
    profile::apply(&mut back, "guess_bonds = maybe\n");
    assert!(back.guess_bonds, "unreadable: the default");
    assert!(search::ENTRIES.iter().any(|e| e.name == "Guess bonds from experience" && e.anchor == "m-bonds"), "a Ctrl K entry");
    // Reset view settings puts it back on (display state: written directly, no action)
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    view_panel(&mut st, &[]);
    st.guess_bonds = false;
    let outs = click_widget(&ctx, &mut st, &inp, "view-reset");
    assert!(st.guess_bonds, "Reset view settings: on again");
    assert!(outs.iter().all(|o| o.actions.is_empty()), "no action for a display setting");
}

/// User request 2026-11-13: while experience draws over the stale bonds, the quiet line over Solve
/// Bonds is experience's (the viewer's `stale_bonds_line`), not the plain "last DFT step" one.
#[test]
fn the_stale_line_is_experiences_when_it_draws() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let stale = bonds_snapshot(&scene, Backend::Estimated, true);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    st.backend = Backend::Estimated;
    let mut inp = input(&scene, Some(&stale), &cam);
    inp.stale_bonds_line = Some("bonds from experience (1 pair) · last DFT t = 12.50 fs".into());
    inp.bonds_label = inp.stale_bonds_line.clone();
    fn walk(s: &egui::epaint::Shape, out: &mut String) {
        match s {
            egui::epaint::Shape::Text(t) => {
                out.push_str(t.galley.text());
                out.push('\n');
            }
            egui::epaint::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
            _ => {}
        }
    }
    let mut text = String::new();
    PROBES.with(|p| p.borrow_mut().clear());
    for _ in 0..3 {
        let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0))), ..Default::default() };
        let full = ctx.run_ui(raw, |root| {
            show(root, &mut st, &inp);
        });
        text.clear();
        full.shapes.iter().for_each(|c| walk(&c.shape, &mut text));
    }
    assert!(PROBES.with(|p| p.borrow().get("bonds-stale-line").is_some()), "the line is drawn");
    assert!(text.contains("bonds from experience (1 pair)") && !text.contains("bonds from the last DFT step"), "{text}");
}

/// User request 2026-11-13: with the bonds stale the 3D view says so in a small line just over the
/// bottom-right "Solve Bonds", right-aligned with it, inside the view and narrower than the
/// bottom-centre label's margin, at a wide and a compact width; the Selection panel keeps its note;
/// a fresh graph has no line; the toasts sit above it.
#[test]
fn the_stale_bonds_line_sits_over_bonds_now() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let stale = bonds_snapshot(&scene, Backend::Estimated, true);
    let fresh = bonds_snapshot(&scene, Backend::Estimated, false);
    for w in [1400.0, 760.0] {
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        st.backend = Backend::Estimated;
        let mut inp = input(&scene, Some(&stale), &cam);
        inp.bonds_label = bonds_label(&stale.dynamics);
        PROBES.with(|p| p.borrow_mut().clear());
        let mut view = None;
        for _ in 0..3 {
            view = frame(&ctx, &mut st, &inp, w, vec![]).view;
        }
        let (line, button) = (probed_rect("bonds-stale-line"), probed_rect("bonds-now"));
        let view = view.expect("the 3D view");
        assert!(line.bottom() <= button.top() + 0.5 && button.top() - line.bottom() < 12.0, "{w}: just over the button: {line:?} {button:?}");
        assert!((line.right() - button.right()).abs() < 2.0, "{w}: right-aligned: {line:?} {button:?}");
        assert!(view.contains_rect(line), "{w}: inside the view {view:?}: {line:?}");
        assert!(line.width() <= overlays::STALE_LINE_WIDTH + 1.0 && line.height() < 50.0, "{w}: small: {line:?}");
        // fresh: no line
        let inp = input(&scene, Some(&fresh), &cam);
        PROBES.with(|p| p.borrow_mut().clear());
        frame(&ctx, &mut st, &inp, w, vec![]);
        assert!(PROBES.with(|p| p.borrow().get("bonds-stale-line").is_none()), "{w}: fresh bonds: no line");
    }
}

/// Partial charges (`sim::charges`): drawn only for the scene's nuclei and when they add up to the
/// scene's charge; current at the geometry they were measured at, else faded with "charges from
/// t = … fs" naming the density; the text carries the sign always and a true minus, near-zero is
/// grey and printed +0.00.
#[test]
fn partial_charges_are_shown_balanced_and_labelled_when_earlier() {
    let scene = Scene::preset("h2").unwrap();
    let mut s = snapshot(&scene, Backend::Estimated, Some(model()));
    assert!(shown_charges(&s.dynamics).is_none() && charges_label(&s.dynamics).is_none(), "none measured");
    let pos: Vec<[f64; 3]> = s.dynamics.scene.nuclei.iter().map(|n| n.pos).collect();
    let mut c = sim::PartialCharges {
        q: vec![0.0004, -0.0004],
        counted: vec![1.0, 1.0],
        z: vec![1, 1],
        positions: pos.clone(),
        electrons: 2.0,
        system_charge: 0,
        t: 12.5 / units::au_time_to_fs(1.0),
        step: 4,
        source: sim::ChargeSource::Step,
        ionic: "none".into(),
        dipole: [0.0; 3],
    };
    s.dynamics.charges = Some(std::sync::Arc::new(c.clone()));
    let (shown, current) = shown_charges(&s.dynamics).expect("balanced, these nuclei");
    assert!(current && shown.q.len() == 2);
    assert_eq!(charges_label(&s.dynamics), None, "measured here");
    // the atoms moved since (a model step): faded, labelled with the density's time and kind
    s.dynamics.scene.nuclei[1].pos[0] += 0.01;
    c.source = sim::ChargeSource::Side(sim::SideKind::Audit);
    s.dynamics.charges = Some(std::sync::Arc::new(c.clone()));
    assert_eq!(shown_charges(&s.dynamics).map(|x| x.1), Some(false));
    assert_eq!(charges_label(&s.dynamics).as_deref(), Some("charges from t = 12.50 fs (an audit's DFT density) — faded until the next DFT density"));
    // a sum that misses the scene's charge is not drawn; nor charges of other nuclei
    s.dynamics.charges = Some(std::sync::Arc::new(sim::PartialCharges { q: vec![1.0, 1.0], ..c.clone() }));
    assert!(shown_charges(&s.dynamics).is_none(), "Σq = 2 against charge 0");
    s.dynamics.charges = Some(std::sync::Arc::new(sim::PartialCharges { z: vec![1], q: vec![0.0], counted: vec![1.0], positions: pos[..1].to_vec(), ..c }));
    assert!(shown_charges(&s.dynamics).is_none(), "one nucleus, the scene has two");
    assert_eq!((charge_text(0.4234), charge_text(-0.4234), charge_text(-0.004)), ("+0.42".to_string(), "\u{2212}0.42".to_string(), "+0.00".to_string()));
    assert!(charge_colour(0.3, 255).b() > charge_colour(0.3, 255).r() && charge_colour(-0.3, 255).r() > charge_colour(-0.3, 255).b());
    assert_eq!(charge_colour(0.001, 255), charge_colour(-0.001, 255), "near zero: grey either way");
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
    use crate::quality::{gpu_solvers, vram_check, VramBudget, VRAM_BUDGET_GB};
    let budget = VramBudget::fixed((VRAM_BUDGET_GB * 1e9) as u64);
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
    assert!(e.contains("GPU memory") && e.contains("the 12.0 GB budget") && e.contains("201³"), "{e}");
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
    for tab in DockTab::PANELS {
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
    [K::Samples, K::ResidualDofE, K::ResidualDofF, K::EnergyResidual, K::ForceResidual, K::Misfit, K::ForceNoise, audit]
        .into_iter()
        .map(|k| TrustCheck { key: k, met: k != key, value: 0.0, limit: 1.0, worst: None })
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
    use crate::quality::VramBudget;
    let a = box_advice_with(&scene, 0, VramBudget::fixed(u64::MAX));
    assert!(a.starts_with("Enlarge the cell (Scene ▸ Grid: 123³ at 0.15 Bohr, a 18.3 Bohr box — every nucleus ≥ 5.3 Bohr from a face"), "{a}");
    assert!(!a.contains("refuse") && !a.contains("beyond"), "{a}");
    assert!(box_advice_with(&scene, 2, VramBudget::fixed(1)).contains("The GPU memory guard would refuse that cell"));
    // already inside that box (the samples reached farther than the scene does now): grow by
    // d_box on every side; past the largest grid: say so
    let mut far = scene.clone();
    far.nuclei[2].pos = [20.0, 0.0, 0.0];
    assert!(box_advice_with(&far, 0, VramBudget::fixed(u64::MAX)).contains("beyond the largest grid (201³)"));
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
            grid_split: None, box_state: false, smeared_1s: false, residual: None, not_converged: false, l: None, touches_box: false, shares: None,
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
            share_z: None,
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
        st.workspace.close_panel(DockTab::Orbitals);
        assert!(!st.tab_shown(DockTab::Orbitals));
        // the View panel's What you see (on screen in the default layout)
        assert!(st.tab_shown(DockTab::View) && st.view_sections.is_open(crate::ui::dock::view::Section::What));
        let outs = click_widget(&ctx, &mut st, &inp, "menu-show-orbitals");
        let sent = orbital_view_commands(&outs);
        assert_eq!(sent.len(), 1, "one request");
        assert_eq!(sent[0].what, OrbitalRequest::Orbital { spin: Spin::Up, track: crate::ui::HOMO_TRACK, squared: false });
        assert_eq!(sent[0].unoccupied, 0, "the eigensolve waits for the first payload (Viewer::drive_orbital_view)");
        assert!(st.unocc_deferred && st.show == ShowMode::Orbitals);
        assert!(st.tab_shown(DockTab::Orbitals), "the Orbitals panel opens");
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
            Action::WarpFields(false),
        ] {
            let text = a.describe().expect("a script form");
            let words: Vec<&str> = text.split(' ').collect();
            assert_eq!(Action::parse(&words).unwrap(), a, "{text}");
        }
        assert!(Action::parse(&["show", "plasma"]).unwrap_err().contains("density, orbitals, colour or spin"));
    }

    /// An O-atom-like list: ↑ s + p×3, ↓ s + p (each ↓ sharing its ↑ partner's slot), one
    /// unoccupied ↑ row; the angular characters known (a lone atom's, pseudised or not).
    fn atom_rows() -> Vec<OrbitalInfo> {
        let with_l = |mut r: OrbitalInfo, l: Option<u8>| {
            r.l = l;
            r
        };
        vec![
            with_l(row(Spin::Up, 0, 1, 0, -0.9, 1.0, idx(1)), Some(0)),
            with_l(row(Spin::Up, 1, 2, 1, -0.34, 1.0, idx(2)), Some(1)),
            with_l(row(Spin::Up, 2, 3, 2, -0.34, 1.0, idx(3)), Some(1)),
            with_l(row(Spin::Up, 3, 4, 3, -0.34, 1.0, idx(4)), Some(1)),
            with_l(row(Spin::Up, 4, 7, 4, 0.05, 0.0, idx(5)), Some(0)),
            with_l(row(Spin::Down, 0, 5, 0, -0.8, 1.0, idx(1)), Some(0)),
            with_l(row(Spin::Down, 1, 6, 1, -0.2, 1.0, idx(2)), Some(1)),
        ]
    }

    /// The quick presets (user request 2026-10-20): each sets the include list from the rows on
    /// screen and the core flag, keeps the draw look, sends the view, is the one highlighted
    /// afterwards, and is a script word.
    #[test]
    fn colour_presets_set_the_include_list_and_round_trip() {
        use crate::ui::actions::{preset_active, preset_include};
        use crate::ui::ColourPreset as P;
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let rows = atom_rows();
        let snap = with_payload(&scene, Backend::Gpu, payload(rows.clone(), OrbitalRequest::List));
        let inp = input(&scene, Some(&snap), &cam);
        let mut st = UiState::new("h2", &scene);
        st.show = ShowMode::Coloured;
        let look = sim::ColourLook { enclose: Some(75), shell: false, hide_core: false };
        act(&mut st, &inp, Action::ColourLook(look));
        assert_eq!(preset_active(&st, &rows), Some(P::Everything), "the default is everything");
        let set = |v: &[u32]| Some(v.iter().copied().collect::<std::collections::BTreeSet<u32>>());
        for (p, include, hide_core) in [
            (P::Valence, None, true),
            (P::OneSpin, set(&[1, 2, 3, 4]), false),
            (P::POnly, set(&[2, 3, 4, 6]), false),
            (P::Everything, None, false),
        ] {
            assert_eq!(preset_include(p, &rows), include, "{p:?}");
            let out = act(&mut st, &inp, Action::ColourPreset(p));
            let r = match out.commands.as_slice() {
                [Command::OrbitalView(r)] => r.clone(),
                c => panic!("{p:?}: {c:?}"),
            };
            assert_eq!(st.included, include, "{p:?}");
            assert_eq!(st.colour_look, sim::ColourLook { hide_core, ..look }, "{p:?}: only the core flag changes");
            assert_eq!(r.what, OrbitalRequest::AllColoured { include: include.map(|s| s.into_iter().collect()), physical: false, look: st.colour_look });
            assert_eq!(preset_active(&st, &rows), Some(p), "{p:?} is highlighted");
        }
        assert_eq!(preset_include(P::POnly, &h2_rows()), set(&[]), "a molecule has no p rows (the button is off there)");
        for p in P::ALL {
            let a = Action::ColourPreset(p);
            let text = a.describe().expect("a script form");
            let words: Vec<&str> = text.split(' ').collect();
            assert_eq!(Action::parse(&words).unwrap(), a, "{text}");
        }
        assert!(Action::parse(&["colour-preset", "d"]).unwrap_err().contains("valence, one-spin, p or everything"));
    }

    /// The coloured legend (user request 2026-10-20): one chip per colour slot — an ↑ orbital and
    /// its ↓ partner are one — naming the channel when a slot is drawn in one only; a ticked
    /// unoccupied orbital has a chip only as a boundary surface; an occupied orbital whose
    /// boundary reaches the box puts the warning under the legend and a tag on its row.
    #[test]
    fn the_coloured_legend_has_one_chip_per_pair_and_warns_of_the_box() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let mut st = UiState::new("h2", &scene);
        st.show = ShowMode::Coloured;
        let colour = Some(sim::OrbitalField::CpuColour { rgba: std::sync::Arc::new(vec![[0.0; 4]; 27]), scale: 1.0 });
        let legend = |rows: Vec<OrbitalInfo>, include: Option<Vec<u32>>, look: sim::ColourLook| {
            let mut o = payload(rows, OrbitalRequest::AllColoured { include, physical: false, look });
            o.field = colour.clone();
            let snap = with_payload(&scene, Backend::Gpu, o);
            legend_of(&st, &input(&scene, Some(&snap), &cam)).unwrap()
        };
        let shell = sim::ColourLook::default();
        let l = legend(atom_rows(), None, shell);
        let names: Vec<&str> = l.swatches.iter().map(|s| s.1.as_str()).collect();
        assert_eq!(names, ["#1", "#2", "#3↑", "#4↑"], "s and the first p are pairs; the other p's are ↑ only");
        assert!(l.text.contains("boundary surface") && l.text.contains("90 %"), "{}", l.text);
        assert!(l.warn.is_none());
        // the unoccupied #5 ticked: a chip as a boundary surface, none by density
        let l = legend(atom_rows(), Some(vec![1, 7]), shell);
        assert_eq!(l.swatches.len(), 2);
        let l = legend(atom_rows(), Some(vec![1, 7]), sim::ColourLook { enclose: None, ..shell });
        assert_eq!(l.swatches.len(), 1, "by density the unoccupied orbital is not drawn");
        // the box
        let mut rows = atom_rows();
        rows[1].touches_box = true;
        rows[6].touches_box = true;
        let l = legend(rows.clone(), None, shell);
        assert_eq!(l.warn.as_deref(), Some("boundary touches the box: enlarge the box (#2)"), "named once per pair");
        let v = RowView::of(&rows[1], &Levels::of(&rows));
        assert!(v.tags.iter().any(|t| t.text == orbitals::TOUCHES_BOX && t.warn), "{:?}", v.tags);
        // an unoccupied row keeps its own box tag instead
        rows[4].touches_box = true;
        assert!(!RowView::of(&rows[4], &Levels::of(&rows)).tags.iter().any(|t| t.text == orbitals::TOUCHES_BOX));
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
        alone(&mut st, DockTab::Orbitals);
        let outs = click_widget(&ctx, &mut st, &inp, "orbital-↓-2");
        let sent = orbital_view_commands(&outs);
        assert_eq!(sent.len(), 1, "{sent:?}");
        assert_eq!(sent[0].what, OrbitalRequest::Orbital { spin: Spin::Down, track: 2, squared: false });
        assert_eq!(st.orbital_pick, Some((Spin::Down, 2)));
        // the tab's own show selector
        let outs = click_widget(&ctx, &mut st, &inp, "show-spin");
        assert_eq!(orbital_view_commands(&outs)[0].what, OrbitalRequest::SpinDensity);
    }

    /// A closed-shell spin readout: ∫|m| inside the solver's noise budget (no net spin).
    fn closed_spin() -> SpinSummary {
        SpinSummary { n_up: 1, n_dn: 1, integral: 0.0, abs_integral: 2e-6, max_abs: 1e-7, rho_max: 0.4, scale: 4e-4, noise: 1.4e-4 }
    }

    /// A closed-shell H₂ list: σ_g and σ_u in both channels (↓ on ↑'s slots), the ↓ energies off by
    /// the solver's spin noise, tracks ↑ 1, 3 and ↓ 2, 4, with a readout that says no net spin.
    fn h2_closed() -> OrbitalDisplay {
        let rows = vec![
            row(Spin::Up, 0, 1, 0, -0.5, 1.0, idx(1)),
            row(Spin::Up, 1, 3, 1, 0.02, 0.0, idx(2)),
            row(Spin::Down, 0, 2, 0, -0.5 + 5e-9, 1.0, idx(1)),
            row(Spin::Down, 1, 4, 1, 0.02 - 5e-9, 0.0, idx(2)),
        ];
        let mut o = payload(rows, OrbitalRequest::List);
        o.spin = closed_spin();
        o
    }

    /// The fold test (user request 2026-11-11): the channels are one closed shell only when every
    /// pair matched by index has the same occupation and ε within the 1e-4 Ha level window, the rows
    /// count the same, and the spin readout says no net spin; the pair's row sums the occupations.
    #[test]
    fn a_closed_shell_pairs_its_channels() {
        let o = h2_closed();
        let closed = closed_spin();
        assert!(orbitals::paired(&o.rows, &closed), "H₂: ↑ and ↓ differ by 5e-9 Ha");
        let mut far = o.rows.clone();
        far[2].energy = -0.5 + 2e-4;
        assert!(!orbitals::paired(&far, &closed), "2e-4 Ha apart: two levels");
        let mut occ = o.rows.clone();
        occ[2].occupation = 0.5;
        assert!(!orbitals::paired(&occ, &closed), "occupations differ");
        // user report 2026-11-12 (He, 4 unoccupied per spin): an unoccupied row only one channel
        // shows still pairs; an occupied one does not
        assert!(orbitals::paired(&o.rows[..3], &closed), "↑ shows σ_u, ↓ does not: still one closed shell");
        assert!(orbitals::paired(&h2_rows(), &closed), "the older fixture: the ↓ σ_u is not shown");
        let no_dn_occ = [o.rows[0].clone(), o.rows[1].clone(), o.rows[3].clone()];
        assert!(!orbitals::paired(&no_dn_occ, &closed), "↓ lacks the occupied σ_g");
        let mut far_unocc = o.rows.clone();
        far_unocc[3].energy = 0.02 + 2e-4;
        assert!(!orbitals::paired(&far_unocc, &closed), "an unoccupied pair 2e-4 Ha apart: two levels");
        let broken = SpinSummary { abs_integral: 0.6, max_abs: 0.1, ..closed };
        assert!(!orbitals::paired(&o.rows, &broken), "a broken-symmetry singlet: same ε, not the same shapes");
        assert!(!orbitals::paired(&o.rows, &SpinSummary::default()), "no noise budget: no verdict");
        let lone_h = [row(Spin::Up, 0, 1, 0, -0.24, 1.0, idx(1))];
        assert!(!orbitals::paired(&lone_h, &SpinSummary { n_dn: 0, integral: 1.0, abs_integral: 1.0, ..closed }));
        assert!(!orbitals::paired(&[], &closed), "no electrons");
        // the pair's row: ε and tags of the ↑ member, the occupations summed
        let l = Levels::of(&o.rows);
        let v = RowView::pair(&o.rows[0], &o.rows[2], &l);
        assert_eq!((v.occupation.as_str(), v.ha.as_str()), ("2", RowView::of(&o.rows[0], &l).ha.as_str()));
        assert!(v.tags.iter().any(|t| t.text == "HOMO"), "{:?}", v.tags);
        let mut third = o.rows.clone();
        third[0].occupation = 1.0 / 3.0;
        third[2].occupation = 1.0 / 3.0;
        assert_eq!(RowView::pair(&third[0], &third[2], &Levels::of(&third)).occupation, "0.67", "a fractional shell sums too");
    }

    /// On screen, a closed shell is one ↑↓ list (user request 2026-11-11): one heading, one row per
    /// pair probed `orbital-↑↓-{↑ track}` reading `occ 2`, no ↓ block; a click picks the ↑ member,
    /// the include box of a pair toggles both tracks; an open shell keeps its two blocks.
    #[test]
    fn a_closed_shell_is_one_paired_list_on_screen() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let snap = with_payload(&scene, Backend::Gpu, h2_closed());
        let inp = input(&scene, Some(&snap), &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        st.show = ShowMode::Orbitals;
        alone(&mut st, DockTab::Orbitals);
        for _ in 0..3 {
            frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        }
        PROBES.with(|p| p.borrow_mut().clear());
        let texts: Vec<String> = frame_texts(&ctx, &mut st, &inp, vec![], 5.0).into_iter().map(|(t, _)| t).collect();
        let has = |needle: &str| texts.iter().any(|t| t.contains(needle));
        assert!(has("↑↓ paired · 1 occupied") && has("↑↓ unoccupied") && has("#1 ↑↓") && has("occ 2"), "{texts:?}");
        assert!(!has("↑ spin") && !has("↓ spin") && !has("↓: none occupied"), "{texts:?}");
        let probed = |name: &str| PROBES.with(|p| p.borrow().contains_key(name));
        assert!(probed("orbital-↑↓-1") && probed("orbital-↑↓-3"));
        assert!(!probed("orbital-↑-1") && !probed("orbital-↓-2"), "no row per member");
        let outs = click_widget(&ctx, &mut st, &inp, "orbital-↑↓-1");
        assert_eq!(orbital_view_commands(&outs)[0].what, OrbitalRequest::Orbital { spin: Spin::Up, track: 1, squared: false });
        assert_eq!(st.orbital_pick, Some((Spin::Up, 1)));
        // All orbitals coloured: the pair's box takes both members out
        st.show = ShowMode::Coloured;
        click_widget(&ctx, &mut st, &inp, "include-1");
        assert_eq!(st.included, Some(std::collections::BTreeSet::new()), "σ_g ↑ (1) and ↓ (2) both off");
        // an open shell (a lone H: one ↑ electron, no ↓) draws as before
        let lone = payload(vec![row(Spin::Up, 0, 1, 0, -0.24, 1.0, idx(1))], OrbitalRequest::List);
        let mut lone = lone;
        lone.spin = SpinSummary { n_dn: 0, integral: 1.0, abs_integral: 1.0, ..closed_spin() };
        let snap = with_payload(&scene, Backend::Gpu, lone);
        let inp = input(&scene, Some(&snap), &cam);
        st.show = ShowMode::Orbitals;
        for _ in 0..3 {
            frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        }
        let texts: Vec<String> = frame_texts(&ctx, &mut st, &inp, vec![], 6.0).into_iter().map(|(t, _)| t).collect();
        let has = |needle: &str| texts.iter().any(|t| t.contains(needle));
        assert!(has("↑ spin · 1 occupied") && has("↓: none occupied") && !has("↑↓ paired"), "{texts:?}");
    }

    /// A folded ↑↓ row never hides a member's caveat (final review M1): a warning only the ↓
    /// member carries (its own display eigensolve stopped at the cap) is drawn on the pair's row
    /// with its arrow; one both carry is drawn once, unmarked; the ↑ member's plain tags stay.
    #[test]
    fn a_folded_pair_shows_the_down_members_warnings() {
        let mut o = h2_closed();
        o.rows[2].not_converged = true;
        o.rows[2].residual = Some(3e-4);
        let l = Levels::of(&o.rows);
        let v = RowView::pair(&o.rows[0], &o.rows[2], &l);
        let texts: Vec<&str> = v.tags.iter().map(|t| t.text.as_str()).collect();
        assert!(texts.contains(&"HOMO"), "the ↑ member's plain tags stay: {texts:?}");
        let nc: Vec<_> = v.tags.iter().filter(|t| t.text.contains("not converged")).collect();
        assert_eq!(nc.len(), 1, "{texts:?}");
        assert_eq!(nc[0].text, "↓ not converged (residual 3.0e-4)");
        assert!(nc[0].warn && nc[0].tip.starts_with("the ↓ orbital of this pair only"), "{:?}", nc[0]);
        // both members not converged: one tag, no arrow
        let mut both = o.rows.clone();
        both[0].not_converged = true;
        both[0].residual = Some(3e-4);
        let v = RowView::pair(&both[0], &both[2], &Levels::of(&both));
        let nc: Vec<_> = v.tags.iter().filter(|t| t.text.contains("not converged")).map(|t| t.text.as_str()).collect();
        assert_eq!(nc, ["not converged (residual 3.0e-4)"], "the same warning is shown once");
        // on screen: the pair's row carries the ↓ member's warning
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let snap = with_payload(&scene, Backend::Gpu, o);
        let inp = input(&scene, Some(&snap), &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        st.show = ShowMode::Orbitals;
        alone(&mut st, DockTab::Orbitals);
        for _ in 0..3 {
            frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        }
        let texts: Vec<String> = frame_texts(&ctx, &mut st, &inp, vec![], 5.0).into_iter().map(|(t, _)| t).collect();
        let has = |needle: &str| texts.iter().any(|t| t.contains(needle));
        assert!(has("#1 ↑↓") && has("↓ not converged (residual 3.0e-4)"), "{texts:?}");
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
            assert!(search::ENTRIES.iter().any(|e| e.name == name && e.place == search::Place::Dock(DockTab::View)), "{name}");
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
        assert_eq!(l.swatches[0].1, "#1", "a pair's chip names no channel");
        assert_eq!(l.text, "each orbital's boundary surface: the region holding 90 % of it, drawn as a shell");
        st.colour_look.enclose = None;
        let mut o = payload(h2_rows(), OrbitalRequest::AllColoured { include: None, physical: false, look: st.colour_look });
        o.field = Some(sim::OrbitalField::CpuColour { rgba: std::sync::Arc::new(vec![[0.0; 4]; 27]), scale: 1.0 });
        let snap = with_payload(&scene, Backend::Gpu, o);
        let inp = input(&scene, Some(&snap), &cam);
        assert_eq!(legend_of(&st, &inp).unwrap().text, "each orbital scaled to its own peak");
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
        alone(&mut st, DockTab::Orbitals);
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
        // a pseudised species has no core on the grid: water under `All` smears nothing, under `CoresOnly` its H's only
        // (M7 Task 13 Step 5 check: the note read "Zσ = 1.2" for the pseudo-O)
        let water = Scene::preset("h2o").unwrap().to_sim().unwrap();
        let sigma = water.sigma.unwrap_or(water.cell.h);
        let summary = |z: u8| sim::PseudoSummary { z, z_v: 1.0, f: 1.2, local: String::new(), r_c_max: 1.0, h_req: None, h_eb: None, r_core: 0.5, record: String::new() };
        assert_eq!(orbitals::largest_smeared_z_sigma(&water, &[]), 8.0 * sigma, "all-electron: the O");
        assert_eq!(orbitals::largest_smeared_z_sigma(&water, &[summary(8)]), sigma, "pseudo-O: the H's");
        assert_eq!(orbitals::largest_smeared_z_sigma(&water, &[summary(8), summary(1)]), 0.0, "all pseudised: none");
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
        // the default look is the boundary surfaces: the dots fill each orbital's region
        let boundary = orbitals::dots_note(&st);
        assert!(boundary.contains("boundary surface") && boundary.contains("not the density"), "{boundary}");
        st.colour_look.enclose = None;
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

    // ---- orbitals by atom (orbital-atom-shares plan Task 4; user request 2026-11-12) ----

    /// `r` with the Hirshfeld shares `s` (one per nucleus, as the provider ships them).
    fn on(mut r: OrbitalInfo, s: &[f64]) -> OrbitalInfo {
        r.shares = Some(s.to_vec());
        r
    }

    fn heads(g: &[orbitals::Group], z: &[u8]) -> Vec<String> {
        g.iter().map(|g| orbitals::heading(&g.owner, g.others, z)).collect()
    }

    /// An H₂O-like list: two rows mostly on O (0.92, 0.60 — the 0.55 line), one spread over all
    /// three atoms (O 0.5, H 0.25, H 0.25), one on H #1. Atoms first by number, then the shared
    /// group; rows keep their order; each row's share names its largest atom, the hover every atom
    /// ≥ 1 % and the Hirshfeld caveat.
    #[test]
    fn groups_h2o_like() {
        let z = [8, 1, 1];
        let rows = vec![
            on(row(Spin::Up, 0, 1, 0, -0.9, 1.0, idx(1)), &[0.92, 0.04, 0.04]),
            on(row(Spin::Up, 1, 2, 1, -0.6, 1.0, idx(2)), &[0.60, 0.20, 0.20]),
            on(row(Spin::Up, 2, 3, 2, -0.5, 1.0, idx(3)), &[0.5, 0.25, 0.25]),
            on(row(Spin::Up, 3, 4, 3, -0.4, 1.0, idx(4)), &[0.05, 0.90, 0.05]),
        ];
        let refs: Vec<&OrbitalInfo> = rows.iter().collect();
        let g = orbitals::groups(&refs, &z, true);
        assert_eq!(heads(&g, &z), ["O #0", "H #1", "shared · O #0–H #1–H #2"]);
        assert_eq!(g[0].rows, [0, 1], "O #0 holds two rows, in their order");
        assert_eq!((g[1].rows.as_slice(), g[2].rows.as_slice()), (&[3][..], &[2][..]));
        assert_eq!(g[2].owner, orbitals::Owner::Shared(vec![0, 1, 2]));
        assert_eq!(orbitals::share_text(&[0.92, 0.04, 0.04], &z), "O #0 92 %");
        assert_eq!(orbitals::share_text(&[0.5, 0.25, 0.25], &z), "O #0 50 %");
        let hover = orbitals::share_hover(&[0.5, 0.25, 0.25], &z);
        assert!(hover.starts_with("O #0 50 % · H #1 25 % · H #2 25 %"), "{hover}");
        assert!(hover.contains("a split of space between the atoms (Hirshfeld, like the partial charges), not something measured"), "{hover}");
        // below 1 % is left out of the hover
        // review Minor 3: in a molecule a share that rounds to 100 % reads "> 99 %"
        assert!(orbitals::share_hover(&[0.995, 0.004, 0.001], &z).starts_with("O #0 > 99 %\n"));
        assert_eq!(orbitals::share_text(&[0.9993, 0.0004, 0.0003], &z), "O #0 > 99 %");
        assert_eq!(orbitals::share_text(&[0.994, 0.003, 0.003], &z), "O #0 99 %");
    }

    /// Near and exact 50/50 read `shared`, never one atom (the 0.55 margin); equal at the whole
    /// percent shown goes by atom number, so the order does not flip with the noise.
    #[test]
    fn an_exact_tie_reads_shared() {
        use orbitals::Owner;
        assert_eq!(orbitals::owner_of(&[0.5000001, 0.4999999]), Owner::Shared(vec![0, 1]));
        assert_eq!(orbitals::owner_of(&[0.4999999, 0.5000001]), Owner::Shared(vec![0, 1]), "a tie at the percent shown: by atom number");
        assert_eq!(orbitals::owner_of(&[0.54, 0.46]), Owner::Shared(vec![0, 1]));
        assert_eq!(orbitals::owner_of(&[0.46, 0.54]), Owner::Shared(vec![1, 0]), "largest first");
        assert_eq!(orbitals::owner_of(&[0.56, 0.44]), Owner::Atom(0));
        assert_eq!(orbitals::owner_of(&[0.44, 0.56]), Owner::Atom(1));
        assert_eq!(orbitals::owner_of(&[0.55, 0.45]), Owner::Atom(0), "at least 0.55");
        assert_eq!(orbitals::share_text(&[0.4999999, 0.5000001], &[1, 1]), "H #0 50 %");
    }

    /// A degenerate set (the provider has already given each member the set's mean) lands in one
    /// group, whatever sits between its members.
    #[test]
    fn degenerate_rows_stay_together() {
        let z = [6, 1, 1, 1, 1];
        let t2 = [0.40, 0.15, 0.15, 0.15, 0.15];
        let set = |mut r: OrbitalInfo| {
            r.cluster = Some((3, 2e-5));
            on(r, &t2)
        };
        let rows = vec![
            on(row(Spin::Up, 0, 1, 0, -9.8, 1.0, idx(1)), &[0.99, 0.0025, 0.0025, 0.0025, 0.0025]),
            set(row(Spin::Up, 1, 2, 1, -0.35, 1.0, idx(2))),
            on(row(Spin::Up, 2, 3, 2, -0.34, 1.0, idx(3)), &[0.05, 0.80, 0.05, 0.05, 0.05]),
            set(row(Spin::Up, 3, 4, 3, -0.35, 1.0, idx(4))),
            set(row(Spin::Up, 4, 5, 4, -0.35, 1.0, idx(5))),
        ];
        let refs: Vec<&OrbitalInfo> = rows.iter().collect();
        let g = orbitals::groups(&refs, &z, true);
        let t2_groups: Vec<&orbitals::Group> = g.iter().filter(|g| g.rows.contains(&1)).collect();
        assert_eq!(t2_groups.len(), 1);
        assert_eq!(t2_groups[0].rows, [1, 3, 4], "{g:?}");
        // ruling (f): the heading counts the atoms it does not name (each H at 15 %)
        assert_eq!(heads(&g, &z), ["C #0", "H #1", "shared · C #0 + 4 others"]);
    }

    /// Text shapes with their colour (a `RichText` colour, else the fallback), drawn in a window
    /// tall enough for the whole list.
    fn coloured_texts(ctx: &egui::Context, st: &mut UiState, inp: &PanelInput<'_>) -> Vec<(String, egui::Color32)> {
        fn walk(s: &egui::epaint::Shape, out: &mut Vec<(String, egui::Color32)>) {
            match s {
                egui::epaint::Shape::Text(t) => {
                    let c = t.galley.job.sections.first().map_or(t.fallback_color, |s| s.format.color);
                    out.push((t.galley.text().to_string(), c));
                }
                egui::epaint::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                _ => {}
            }
        }
        let mut last = Vec::new();
        for _ in 0..3 {
            let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 2400.0))), ..Default::default() };
            let mut full = ctx.run_ui(raw, |root| {
                show(root, st, inp);
            });
            full.textures_delta.clear();
            last.clear();
            full.shapes.iter().for_each(|c| walk(&c.shape, &mut last));
        }
        last
    }

    /// A water-like open-shell payload with shares on the scene's three atoms: ↑ #1 O, ↑ #2 O,
    /// ↑ #3 H #1, ↓ #1 O, ↓ #2 H #1 (tracks 1–5).
    fn water_by_atom() -> OrbitalDisplay {
        let rows = vec![
            on(row(Spin::Up, 0, 1, 0, -0.9, 1.0, idx(1)), &[0.98, 0.01, 0.01]),
            on(row(Spin::Up, 1, 2, 1, -0.5, 1.0, idx(2)), &[0.60, 0.20, 0.20]),
            on(row(Spin::Up, 2, 3, 2, -0.4, 1.0, idx(3)), &[0.15, 0.80, 0.05]),
            on(row(Spin::Down, 0, 4, 0, -0.85, 1.0, idx(1)), &[0.98, 0.01, 0.01]),
            on(row(Spin::Down, 1, 5, 2, -0.35, 1.0, idx(2)), &[0.15, 0.80, 0.05]),
        ];
        let mut o = payload(rows, OrbitalRequest::List);
        o.positions = vec![[0.0; 3]; 3];
        o.share_z = Some(vec![8, 1, 1]);
        o
    }

    /// Goal 3 (user decision 2026-11-12): selecting H #1 keeps its rows at full strength and draws
    /// the O rows faint, still in the list (the same rows, the same order) and still clickable;
    /// the Edit tool's group counts as selected; deselecting leaves nothing faint.
    #[test]
    fn selection_dims_but_keeps_rows() {
        let scene = Scene::preset("h2o").unwrap();
        let cam = camera();
        let snap = with_payload(&scene, Backend::Gpu, water_by_atom());
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2o", &scene);
        st.show = ShowMode::Orbitals;
        alone(&mut st, DockTab::Orbitals);
        let names = ["#1 ↑", "#2 ↑", "#3 ↑", "#1 ↓", "#2 ↓"];
        // the row labels on screen, in order, each with whether it is drawn faint
        let rows_drawn = |st: &mut UiState, inp: &PanelInput<'_>| -> Vec<(String, bool)> {
            PROBES.with(|p| p.borrow_mut().clear());
            let t = coloured_texts(&ctx, st, inp);
            let probed = PROBES.with(|p| p.borrow().keys().filter(|k| k.starts_with("orbital-↑-") || k.starts_with("orbital-↓-")).count());
            assert_eq!(probed, 5, "every row is drawn");
            t.into_iter().filter(|(s, _)| names.contains(&s.as_str())).map(|(s, c)| (s, c == theme::INK_FAINT)).collect()
        };
        let none = input(&scene, Some(&snap), &cam);
        let before = rows_drawn(&mut st, &none);
        assert_eq!(before.len(), 5, "{before:?}");
        assert!(before.iter().all(|(_, faint)| !faint), "nothing selected: nothing faint ({before:?})");
        let h1 = PanelInput { selected: Some(1), ..input(&scene, Some(&snap), &cam) };
        let sel = rows_drawn(&mut st, &h1);
        let order = |v: &[(String, bool)]| v.iter().map(|(s, _)| s.clone()).collect::<Vec<_>>();
        assert_eq!(order(&sel), order(&before), "the list does not re-sort on a selection");
        let faint: Vec<&str> = sel.iter().filter(|(_, f)| *f).map(|(s, _)| s.as_str()).collect();
        assert_eq!(faint, ["#1 ↑", "#2 ↑", "#1 ↓"], "the O rows are faint, the H #1 rows lit");
        // a faint row is still a picker
        let outs = click_widget(&ctx, &mut st, &h1, "orbital-↑-1");
        assert!(outs.iter().flat_map(|o| &o.actions).any(|a| *a == Action::PickOrbital { spin: Spin::Up, track: 1 }));
        assert_eq!(st.orbital_pick, Some((Spin::Up, 1)));
        // the Edit tool's group selects too: O #0 and H #1 together light every row here
        st.edit_group = vec![0, 1];
        assert!(rows_drawn(&mut st, &none).iter().all(|(_, f)| !f));
        st.edit_group = vec![2];
        assert_eq!(rows_drawn(&mut st, &none).iter().filter(|(_, f)| *f).count(), 5, "nothing is mostly on H #2");
        st.edit_group.clear();
        assert!(rows_drawn(&mut st, &none).iter().all(|(_, f)| !f), "deselected: nothing faint");
        assert_eq!(orbitals::selected_atoms(Some(1), &[1, 2, 7], 3), [1, 2], "each once, within the scene");
        assert!(orbitals::lit(&[0.3, 0.3, 0.4], &[]) && orbitals::lit(&[0.3, 0.3, 0.4], &[0, 1]) && !orbitals::lit(&[0.3, 0.3, 0.4], &[2]));
    }

    /// A state of the box and an unoccupied row of an earlier geometry are listed under their own
    /// headings and show no share; the occupied rows keep theirs.
    #[test]
    fn box_and_earlier_rows_have_no_share() {
        use orbitals::Owner;
        let z = [8, 1, 1];
        let mut boxed = on(row(Spin::Up, 3, 6, 3, 0.03, 0.0, idx(4)), &[0.4, 0.3, 0.3]); // shares ignored
        boxed.box_state = true;
        let unocc = row(Spin::Up, 4, 7, 4, 0.05, 0.0, idx(5));
        let occ = on(row(Spin::Up, 0, 1, 0, -0.9, 1.0, idx(1)), &[0.98, 0.01, 0.01]);
        let rows = [&occ, &boxed, &unocc];
        let g = orbitals::groups(&rows, &z, false);
        assert_eq!(g.iter().map(|g| g.owner.clone()).collect::<Vec<_>>(), [Owner::Atom(0), Owner::Box, Owner::Earlier]);
        assert_eq!(heads(&g, &z), ["O #0", "box states", "from an earlier geometry"]);
        let current = orbitals::groups(&rows, &z, true);
        assert_eq!(current[2].owner, Owner::Unknown, "of this geometry without shares: unknown, not earlier");
        // on screen: the headings, and a share on the occupied row only
        let scene = Scene::preset("h2o").unwrap();
        let cam = camera();
        let mut o = water_by_atom();
        o.rows.push(boxed.clone());
        o.rows.push(unocc.clone());
        o.unoccupied = Some(UnoccupiedStatus { requested: 4, current: false, ..Default::default() });
        let snap = with_payload(&scene, Backend::Gpu, o);
        let inp = input(&scene, Some(&snap), &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2o", &scene);
        st.show = ShowMode::Orbitals;
        alone(&mut st, DockTab::Orbitals);
        PROBES.with(|p| p.borrow_mut().clear());
        let texts: Vec<String> = coloured_texts(&ctx, &mut st, &inp).into_iter().map(|(t, _)| t).collect();
        for h in ["O #0", "H #1", "box states", "from an earlier geometry"] {
            assert!(texts.iter().any(|t| t == h), "{h}: {texts:?}");
        }
        let probed = |name: &str| PROBES.with(|p| p.borrow().contains_key(name));
        assert!(probed("share-↑-1") && probed("orbital-↑-6") && probed("orbital-↑-7"));
        assert!(!probed("share-↑-6") && !probed("share-↑-7"), "no share for a box state or an earlier geometry's row");
        assert!(texts.iter().any(|t| t == "O #0 98 %"), "{texts:?}");
    }

    /// Review Focus 5: H₂ in play — the σ_g's shares 0.5 ± 1e-6 of grid noise on every step —
    /// reads `shared` with the same heading and the same share every time.
    #[test]
    fn play_keeps_h2_shared() {
        let z = [1, 1];
        let mut seed = 0x9e37_79b9_u32;
        for step in 0..20 {
            // a small LCG: a deterministic ± noise of up to 1e-6
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let e = (seed as f64 / u32::MAX as f64 - 0.5) * 2e-6;
            let mut o = h2_closed();
            o.rows[0].shares = Some(vec![0.5 + e, 0.5 - e]);
            o.rows[2].shares = Some(vec![0.5 + e, 0.5 - e]);
            let [up, dn] = [Spin::Up, Spin::Down].map(|s| o.rows.iter().filter(|r| r.spin == s && r.occupation > 0.0).collect::<Vec<_>>());
            for rows in [up, dn] {
                let g = orbitals::groups(&rows, &z, true);
                assert_eq!(g.len(), 1);
                assert_eq!(g[0].owner, orbitals::Owner::Shared(vec![0, 1]), "step {step}: e {e:e}");
                assert_eq!(heads(&g, &z), ["shared · H #0–H #1"], "step {step}");
            }
            assert_eq!(orbitals::share_text(o.rows[0].shares.as_ref().unwrap(), &z), "H #0 50 %", "step {step}");
        }
    }

    /// Spec goal 4: a lone atom lists exactly as without shares — no group heading, no share.
    #[test]
    fn one_nucleus_lists_as_today() {
        let scene = Scene::preset("he").unwrap();
        let cam = camera();
        let he_rows = || vec![row(Spin::Up, 0, 1, 0, -0.57, 1.0, idx(1)), row(Spin::Down, 0, 2, 0, -0.57, 1.0, idx(1))];
        let mut with = payload(he_rows().into_iter().map(|r| on(r, &[1.0])).collect(), OrbitalRequest::List);
        with.share_z = Some(vec![2]);
        with.spin = closed_spin();
        let mut without = payload(he_rows(), OrbitalRequest::List);
        without.spin = closed_spin();
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("he", &scene);
        st.show = ShowMode::Orbitals;
        alone(&mut st, DockTab::Orbitals);
        let mut draw = |o: OrbitalDisplay, sel: Option<usize>| {
            let snap = with_payload(&scene, Backend::Gpu, o);
            let inp = PanelInput { selected: sel, ..input(&scene, Some(&snap), &cam) };
            PROBES.with(|p| p.borrow_mut().clear());
            let t = coloured_texts(&ctx, &mut st, &inp);
            assert!(PROBES.with(|p| !p.borrow().keys().any(|k| k.starts_with("share-"))), "no share column");
            t
        };
        let a = draw(with.clone(), None);
        assert!(!a.iter().any(|(t, _)| t.contains("He #0") || t.contains("shared ·")), "{a:?}");
        assert_eq!(a, draw(without, None), "the same list as without shares");
        // the list (from its heading to the view's status chip): the rest of the chrome shows the selection
        let list = |t: &[(String, egui::Color32)]| {
            let from = t.iter().position(|(s, _)| s == "ORBITALS").expect("the list's heading");
            let to = from + t[from..].iter().position(|(s, _)| s.starts_with("paused")).expect("the status chip");
            t[from..to].to_vec()
        };
        assert!(list(&a).iter().any(|(s, _)| s == "#1 ↑↓"));
        assert_eq!(list(&draw(with, Some(0))), list(&a), "selecting the atom changes nothing in the list");
    }

    /// Controller ruling (Task 4): shares are used only when their Z list is the scene's — a late
    /// payload from before an edit lists as without shares and says why; so does a payload whose
    /// shares failed their check, or one waiting for the free-atom references.
    #[test]
    fn shares_need_the_scenes_atoms() {
        let scene_z = [8, 1, 1];
        let o = water_by_atom();
        assert!(orbitals::grouped(&o, &scene_z) && orbitals::shares_note(&o, &scene_z, false).is_none());
        // an H added since: the payload's shares index [8, 1, 1], the scene is [8, 1, 1, 1]
        let edited = [8, 1, 1, 1];
        assert!(!orbitals::grouped(&o, &edited));
        assert!(orbitals::shares_note(&o, &edited, false).unwrap().contains("before the last edit"));
        let mut bare = water_by_atom();
        bare.share_z = None;
        bare.rows.iter_mut().for_each(|r| r.shares = None);
        assert_eq!(orbitals::shares_note(&bare, &scene_z, true).as_deref(), Some(orbitals::SHARES_PENDING));
        assert!(orbitals::shares_note(&bare, &scene_z, false).unwrap().starts_with("atom shares need the atom references"));
        bare.notes.push("orbital shares off: Σ = 1.002500 outside 1 ± 1e-3".into());
        // review Minor 4: the producer's note is drawn above with the payload's notes; this one
        // points at it instead of repeating it
        let n = orbitals::shares_note(&bare, &scene_z, true).unwrap();
        assert!(n.contains("failed their own check") && !n.contains("1.002500"), "{n}");
        assert!(orbitals::shares_note(&bare, &[2], true).is_none(), "a lone atom: no note");
    }

    /// Ruling (f): a shared heading names the atoms holding ≥ 25 % and counts the others holding
    /// ≥ 1 % of one of its rows ("+ 1 other", "+ k others"), or reads "spread over k atoms" when
    /// none holds 25 %. The count is for display: a tail crossing 1 % never splits the group.
    #[test]
    fn shared_headings_count_the_other_atoms() {
        let z = [6, 1, 1, 1, 1];
        let refs = |rows: &[OrbitalInfo]| -> Vec<orbitals::Group> { orbitals::groups(&rows.iter().collect::<Vec<_>>(), &z, true) };
        // five atoms at 20 % each: nobody named
        let spread = [on(row(Spin::Up, 0, 1, 0, -0.5, 1.0, idx(1)), &[0.2; 5])];
        let g = refs(&spread);
        assert_eq!(g[0].owner, orbitals::Owner::Shared(vec![]));
        assert_eq!(heads(&g, &z), ["shared · spread over 5 atoms"]);
        // one other atom; the same named atoms with and without a tail above 1 %: one group
        let rows = [
            on(row(Spin::Up, 0, 1, 0, -0.6, 1.0, idx(1)), &[0.50, 0.45, 0.05, 0.0, 0.0]),
            on(row(Spin::Up, 1, 2, 1, -0.5, 1.0, idx(2)), &[0.52, 0.48, 0.0, 0.004, 0.0]),
        ];
        let g = refs(&rows);
        assert_eq!(g.len(), 1, "{g:?}");
        assert_eq!((g[0].rows.as_slice(), g[0].others), (&[0, 1][..], 1));
        assert_eq!(heads(&g, &z), ["shared · C #0–H #1 + 1 other"]);
        // no other atom at 1 %: no suffix
        let rows = [on(row(Spin::Up, 0, 1, 0, -0.6, 1.0, idx(1)), &[0.50, 0.496, 0.004, 0.0, 0.0])];
        assert_eq!(heads(&refs(&rows), &z), ["shared · C #0–H #1"]);
        assert!(heads(&refs(&rows), &z)[0].starts_with("shared · ") && orbitals::heading(&orbitals::Owner::Atom(0), 3, &z) == "C #0", "an atom's heading has no count");
    }

    /// Review Minor 2: the 0.55 and 0.25 lines are judged at the whole percent the row prints, so a
    /// row reading "55 %" is never under shared and an atom whose hover reads "25 %" is named.
    #[test]
    fn thresholds_are_judged_at_the_percent_shown() {
        use orbitals::Owner;
        assert_eq!(orbitals::share_text(&[0.546, 0.454], &[8, 1]), "O #0 55 %");
        assert_eq!(orbitals::owner_of(&[0.546, 0.454]), Owner::Atom(0), "prints 55 %: under the atom");
        assert_eq!(orbitals::owner_of(&[0.544, 0.456]), Owner::Shared(vec![0, 1]), "prints 54 %: shared");
        assert_eq!(orbitals::owner_of(&[0.508, 0.246, 0.246]), Owner::Shared(vec![0, 1, 2]), "0.246 prints 25 %: named");
        assert_eq!(orbitals::owner_of(&[0.51, 0.244, 0.246]), Owner::Shared(vec![0, 2]), "0.244 prints 24 %: not named");
        assert!(orbitals::lit(&[0.546, 0.454], &[0]) && !orbitals::lit(&[0.544, 0.456], &[0]));
    }

    /// A closed-shell water-like payload (↑↓ paired) with shares on the scene's atoms: O 1s, an O
    /// lone pair, an O–H bond on H #1 whose ↓ member carries no shares, and an unoccupied row on
    /// H #1 (tracks ↑ 1, 3, 5, 7; ↓ 2, 4, 6, 8).
    fn water_closed_by_atom() -> OrbitalDisplay {
        let pair = |index: usize, track: u32, e: f64, occ: f64, s: &[f64]| {
            [on(row(Spin::Up, index, track, index as u8, e, occ, idx(index + 1)), s), on(row(Spin::Down, index, track + 1, index as u8, e + 5e-9, occ, idx(index + 1)), s)]
        };
        let mut rows: Vec<OrbitalInfo> = Vec::new();
        rows.extend(pair(0, 1, -0.9, 1.0, &[0.996, 0.002, 0.002]));
        rows.extend(pair(1, 3, -0.5, 1.0, &[0.70, 0.15, 0.15]));
        let mut bond = pair(2, 5, -0.4, 1.0, &[0.15, 0.80, 0.05]);
        bond[1].shares = None; // the pair is placed and labelled by its ↑ member
        rows.extend(bond);
        rows.extend(pair(3, 7, 0.05, 0.0, &[0.15, 0.79, 0.06]));
        let mut o = payload(rows, OrbitalRequest::List);
        o.spin = closed_spin();
        o.positions = vec![[0.0; 3]; 3];
        o.share_z = Some(vec![8, 1, 1]);
        o
    }

    /// Review I2: the common case — a closed-shell molecule — grouped by atom keeps the ↑↓ fold in
    /// each group: one pair row and one share per pair, the "↑↓ unoccupied" heading inside the
    /// group holding the unoccupied pair, a pair placed by its ↑ member's shares (its ↓ has none),
    /// the O pairs faint under H #1, a faint pair still a picker, and a picked pair never faint.
    #[test]
    fn grouped_closed_shell_keeps_pairs() {
        let scene = Scene::preset("h2o").unwrap();
        let cam = camera();
        let o = water_closed_by_atom();
        assert!(orbitals::paired(&o.rows, &o.spin));
        let snap = with_payload(&scene, Backend::Gpu, o.clone());
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2o", &scene);
        st.show = ShowMode::Orbitals;
        alone(&mut st, DockTab::Orbitals);
        let none = input(&scene, Some(&snap), &cam);
        PROBES.with(|p| p.borrow_mut().clear());
        let texts = coloured_texts(&ctx, &mut st, &none);
        let count = |prefix: &str| PROBES.with(|p| p.borrow().keys().filter(|k| k.starts_with(prefix)).count());
        assert_eq!((count("orbital-↑↓-"), count("share-↑↓-")), (4, 4), "one row and one share per pair");
        assert_eq!(count("orbital-↑-") + count("orbital-↓-"), 0, "no row per member");
        for t in [1, 3, 5, 7] {
            assert!(PROBES.with(|p| p.borrow().contains_key(&format!("share-↑↓-{t}"))), "share of pair {t}");
        }
        let at = |needle: &str| texts.iter().position(|(t, _)| t == needle).unwrap_or_else(|| panic!("{needle}: {texts:?}"));
        // O #0 holds #1, #2; H #1 holds #3 and, under its own unoccupied heading, #4
        assert!(at("O #0") < at("#1 ↑↓") && at("#2 ↑↓") < at("H #1") && at("H #1") < at("#3 ↑↓"), "{texts:?}");
        assert!(at("#3 ↑↓") < at("↑↓ unoccupied") && at("↑↓ unoccupied") < at("#4 ↑↓"), "the unoccupied heading inside H #1's group");
        assert!(texts.iter().any(|(t, _)| t == "H #1 80 %"), "the ↑ member's share, though the ↓ has none");
        assert!(texts.iter().any(|(t, _)| t == "O #0 > 99 %"));
        // H #1 selected: the O pairs faint, the H #1 pairs lit
        let h1 = PanelInput { selected: Some(1), ..input(&scene, Some(&snap), &cam) };
        let faint = |texts: &[(String, egui::Color32)]| -> Vec<String> {
            texts.iter().filter(|(t, c)| t.ends_with(" ↑↓") && *c == theme::INK_FAINT).map(|(t, _)| t.clone()).collect()
        };
        assert_eq!(faint(&coloured_texts(&ctx, &mut st, &h1)), ["#1 ↑↓", "#2 ↑↓"]);
        // a faint pair is still a picker: the ↑ member
        let outs = click_widget(&ctx, &mut st, &h1, "orbital-↑↓-1");
        assert!(outs.iter().flat_map(|o| &o.actions).any(|a| *a == Action::PickOrbital { spin: Spin::Up, track: 1 }));
        // the payload answering that pick: the picked pair keeps its full ink (review Minor 5)
        let mut picked = o;
        picked.what = OrbitalRequest::Orbital { spin: Spin::Up, track: 1, squared: false };
        let snap = with_payload(&scene, Backend::Gpu, picked);
        let h1 = PanelInput { selected: Some(1), ..input(&scene, Some(&snap), &cam) };
        assert_eq!(faint(&coloured_texts(&ctx, &mut st, &h1)), ["#2 ↑↓"], "picked and not lit: still readable");
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

    /// User request 2026-11-11: while playing, the Activity tab's groups below the job list stay put
    /// as each step's solve starts (no iterations yet), runs (its line and sparkline) and ends (no
    /// row between steps); paused and idle the list shrinks back to "nothing running".
    #[test]
    fn the_job_list_keeps_its_height_between_steps() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let snap = snapshot(&scene, Backend::Gpu, None);
        let live = sim::ScfLive {
            attempt: 1,
            iteration: 6,
            max_iter: 200,
            delta_rho: 5.2e-4,
            tol: 1e-5,
            energy: -1.13,
            d_energy: 5.2e-8,
            ms_per_iteration: 69.0,
            since_gain: 1,
            window: 50,
            best_delta_rho: 5.2e-4,
            stall_level: 8e-3,
            spark: [-3.0; sim::activity::SPARK_POINTS],
            spark_len: 6,
            spark_stride: 1,
        };
        let slot = |running, step, live| sim::ProbeView { main: sim::activity::SlotView { running, scf_step: step, elapsed_s: 0.1, solves: 1, live }, ..Default::default() };
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        st.open_tab(DockTab::Activity);
        let mut counters_at = |playing, probe| {
            let inp = PanelInput { playing, probe: Some(probe), ..input(&scene, Some(&snap), &cam) };
            for _ in 0..2 {
                frame(&ctx, &mut st, &inp, 1400.0, vec![]);
            }
            probed_rect("anchor-d-counters").top()
        };
        let starting = counters_at(true, slot(true, 1, None));
        let solving = counters_at(true, slot(true, 6, Some(live)));
        let between = counters_at(true, slot(false, 0, None));
        let again = counters_at(true, slot(true, 1, None));
        assert!((starting - solving).abs() < 0.5, "a step's first iteration keeps the solve block: {starting} vs {solving}");
        assert!((between - solving).abs() < 0.5 && (again - solving).abs() < 0.5, "between steps the list keeps its height: {between}, {again} vs {solving}");
        let idle = counters_at(false, slot(false, 0, None));
        assert!(idle < solving - 20.0, "paused and idle the list shrinks: {idle} vs {solving}");
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

    /// The display eigensolve (user request 2026-10-19): a job row while it runs, and a log row
    /// with its time merged by time among the teaching rows; the "orbitals" chip hides them; it
    /// counts as no teaching solve.
    #[test]
    fn the_display_eigensolve_has_a_job_row_and_log_rows() {
        use activity::jobs::{rows, JobKey};
        use activity::log::{display_solve_cell, items, Item};
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let snap = snapshot(&scene, Backend::Gpu, None);
        let probe = sim::ProbeView { display: Some((0.3, 4)), display_solves: 2, ..Default::default() };
        let inp = PanelInput { probe: Some(probe), ..input(&scene, Some(&snap), &cam) };
        let r = rows(&inp, inp.probe.as_ref());
        let job = r.iter().find(|r| r.key == JobKey::Orbitals).expect("the eigensolve's job row");
        assert!(job.detail.contains("4 unoccupied orbitals") && job.elapsed_s == Some(0.3));
        assert!(job.control.is_none() && job.no_control_why.is_some());
        let idle = PanelInput { probe: Some(sim::ProbeView::default()), ..input(&scene, Some(&snap), &cam) };
        assert!(rows(&idle, idle.probe.as_ref()).iter().all(|r| r.key != JobKey::Orbitals));

        let mut st = activity::ActivityState::new();
        let solve = |ms: f64, interrupted: bool| sim::DisplaySolve { at: Instant::now(), ms, unoccupied: 4, interrupted };
        st.absorb_display(vec![solve(120.0, false)]);
        std::thread::sleep(Duration::from_millis(2));
        let mut s = snapshot(&scene, Backend::Estimated, Some(model()));
        s.activity.events = vec![event(TeachOrigin::Fallback, TeachRule::Grade, TeachOutcome::Kept, Some(1))];
        st.absorb(&s);
        std::thread::sleep(Duration::from_millis(2));
        st.absorb_display(vec![solve(12_345.0, true)]);
        assert_eq!(st.session_solves, 1, "a display eigensolve is not a teaching solve");
        assert_eq!(items(&st), vec![Item::Display(0), Item::Row(0), Item::Display(1)]);
        assert_eq!(display_solve_cell(&st.display_rows[0].solve), "120 ms");
        assert_eq!(display_solve_cell(&st.display_rows[1].solve), "12.3 s · stopped");
        st.filter.orbitals = false;
        assert_eq!(items(&st), vec![Item::Row(0)]);
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
        // hiding the panel behind another tab of its leaf turns them off again, showing it on
        st.open_tab(DockTab::Graph);
        assert!(matches!(frame(&ctx, &mut st, &inp, 1400.0, vec![]).commands.as_slice(), [Command::SetActivityViews(false)]));
        st.open_tab(DockTab::Activity);
        assert!(matches!(frame(&ctx, &mut st, &inp, 1400.0, vec![]).commands.as_slice(), [Command::SetActivityViews(true)]));
        // and closing it, off
        st.workspace.close_panel(DockTab::Activity);
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
        assert_eq!(rule_sentence(R::Untrusted(TrustKey::Misfit), None, RowCtx::default()), "the model was not trusted then: too many training samples it could not reproduce");
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
        assert_eq!(outcome_label(K::HeldOut), "not kept: held-out check");
        assert_eq!(outcome_sentence(K::HeldOut), "a DFT solve held out on purpose to test the teaching — never trained on");
        assert_eq!(outcome_label(K::Discarded), "discarded");
    }

    /// Every rule and outcome has a label and a sentence (an exhaustive match stops compiling
    /// when a variant is added; this keeps the words non-empty).
    #[test]
    fn every_rule_and_outcome_is_worded() {
        use activity::words::{outcome_label, outcome_sentence, rule_label, rule_sentence, RowCtx};
        for r in [sim::activity::TeachRule::NonFiniteAnswer, sim::activity::TeachRule::NearQuarantine, sim::activity::TeachRule::NearMisfit, sim::activity::TeachRule::Audit, sim::activity::TeachRule::RelaxVerification] {
            assert!(!rule_label(r, None).is_empty() && !rule_sentence(r, None, RowCtx::default()).is_empty(), "{r:?}");
        }
        for o in [sim::activity::TeachOutcome::Failed, sim::activity::TeachOutcome::NotSolved, sim::activity::TeachOutcome::KeptReleased { released: 2, superseded: 1 }, sim::activity::TeachOutcome::MisfitDisagrees] {
            assert!(!outcome_label(o).is_empty() && !outcome_sentence(o).is_empty(), "{o:?}");
        }
        // the misfit fence's words (spec 2026-11-13 §4.3): the refusal is never silent
        assert_eq!(rule_label(sim::activity::TeachRule::NearMisfit, None), "misfit");
        assert_eq!(rule_sentence(sim::activity::TeachRule::NearMisfit, Some(0), RowCtx::default()), "an atom here is next to a training sample the model cannot reproduce");
        assert_eq!(outcome_label(sim::activity::TeachOutcome::MisfitDisagrees), "not kept: misfit");
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
        st.workspace.close_panel(DockTab::Activity);
        click_widget(&ctx, &mut st, &inp, "activity-bar");
        assert!(st.tab_shown(DockTab::Activity), "the bar shows the Activity panel");
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

    /// Spec 2026-11-13 §4.3: the misfit row reads as the training samples the model cannot
    /// reproduce, the count against 10 % of the fitted samples.
    #[test]
    fn the_misfit_line_counts_the_samples_it_cannot_reproduce() {
        use activity::words::{trust_line, AuditSeen};
        use sim::activity::{TrustCheck, TrustKey as K};
        let c = TrustCheck { key: K::Misfit, met: true, value: 3.0, limit: 12.3, worst: None };
        assert_eq!(trust_line(&c, AuditSeen::Never, 20), "Few training samples it cannot reproduce — 3 of at most 12.3 (10 % of the fitted samples)");
    }

    /// Plan review M1 and the Task 3 review: the audit line is per atom or names forces, and when
    /// met it says what the last audit was — or that none ran this fit, or ever — never "passed"
    /// by default.
    #[test]
    fn the_audit_line_says_what_ran() {
        use activity::words::{trust_line, AuditSeen};
        use sim::activity::{TrustCheck, TrustKey as K};
        use sim::AuditVerdict as V;
        let met = TrustCheck { key: K::Audit, met: true, value: 0.0, limit: 1e-3, worst: None };
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
        let energy = TrustCheck { key: K::Audit, met: false, value: 2.5e-3, limit: 1e-3, worst: None };
        assert_eq!(trust_line(&energy, AuditSeen::ThisFit(&rec(V::Failed)), 20), "Last check against DFT — found the energy off by 2.5 mHa/atom, allowed 1 mHa/atom (until a refit predicts it without that sample)");
        let forces = TrustCheck { key: K::AuditForces, met: false, value: 4.2e-3, limit: 2e-3, worst: None };
        assert_eq!(trust_line(&forces, AuditSeen::ThisFit(&rec(V::Failed)), 20), "Last check against DFT — found forces off by 4.2, allowed 2 mHa/Bohr (until a refit predicts it without that sample)");
        // the three states come from the status
        let m = model();
        assert!(matches!(activity::words::AuditSeen::of(&m), AuditSeen::ThisFit(_)));
        assert!(matches!(activity::words::AuditSeen::of(&ModelStatus { last_audit_this_fit: false, ..model() }), AuditSeen::EarlierFit(_)));
        assert!(matches!(activity::words::AuditSeen::of(&ModelStatus { last_audit: None, last_audit_this_fit: false, ..model() }), AuditSeen::Never));
    }

    /// 2026-11-13: the training-residual lines name the sample (and for the forces the atom) that
    /// holds the worst value — store index and formula as the Worst-fitting rows show them, the
    /// origin on hover — and a failed one whose sample Forget would take carries the Forget link
    /// (the Model tab's line too). Without a location the lines read exactly as before.
    #[test]
    fn the_residual_lines_name_the_worst_sample_and_atom() {
        use activity::words::{trust_line, AuditSeen};
        use sim::activity::{TrustCheck, TrustKey as K, TrustWorst};
        let id = sim::SampleId { index: 1234, key: 0xabc };
        let at = |atom, forgettable| Some(TrustWorst { id, formula: "H3O".into(), origin: "fallback".into(), atom, forgettable });
        let force = TrustCheck { key: K::ForceResidual, met: false, value: 3.4e-3, limit: 2e-3, worst: at(Some((2, 8)), true) };
        assert_eq!(trust_line(&force, AuditSeen::Never, 20), "Reproduces its own training forces — worst 3.4 in #1234 · H3O (atom 2, O), allowed 2 mHa/Bohr for that atom");
        let energy = TrustCheck { key: K::EnergyResidual, met: false, value: 2.2e-3, limit: 1e-3, worst: at(None, true) };
        assert_eq!(trust_line(&energy, AuditSeen::Never, 20), "Reproduces its own training energies to the accuracy it claims — worst 2.2 mHa/atom in #1234 · H3O, allowed 1 mHa/atom");
        // the fallback: no location (an old status, no model, a changed sample) — today's text
        let old_f = TrustCheck { worst: None, ..force.clone() };
        assert_eq!(trust_line(&old_f, AuditSeen::Never, 20), "Reproduces its own training forces — worst 3.4, allowed 2 mHa/Bohr for that atom");
        let old_e = TrustCheck { worst: None, ..energy.clone() };
        assert_eq!(trust_line(&old_e, AuditSeen::Never, 20), "Reproduces its own training energies to the accuracy it claims — worst 2.2 mHa/atom, allowed 1 mHa/atom");
        // the Forget link: a failed check whose sample is code 0 only
        assert_eq!(activity::trust::forget_target(&force), Some(id));
        assert_eq!(activity::trust::forget_target(&TrustCheck { worst: at(Some((2, 8)), false), ..force.clone() }), None, "held out: Forget would refuse it");
        assert_eq!(activity::trust::forget_target(&TrustCheck { met: true, ..force.clone() }), None, "a met check blocks nothing");
        assert_eq!(activity::trust::forget_target(&old_f), None);
        // the checklist and the Model tab's refusal line
        let checks = |f: TrustCheck| {
            let mut v = checks_failing(K::ForceResidual);
            let k = v.iter().position(|c| c.key == K::ForceResidual).unwrap();
            v[k] = f;
            v
        };
        let m = ModelStatus { trusted: false, untrusted_reason: Some("max training force residual 3.4e-3 > 2.0e-3 Ha/Bohr".into()), quarantined: 0, box_affected: 0, readiness: None, trust_checks: checks(force.clone()), ..model() };
        let scene = Scene::preset("h2").unwrap();
        let st = UiState::new("h2", &scene);
        let lines = activity::trust::lines(&m, &scene, &st);
        let l = lines.iter().find(|l| l.why).expect("this is why");
        assert_eq!(l.text, trust_line(&force, AuditSeen::Never, 20));
        assert!(l.hover.ends_with("; worst in #1234 · H3O · fallback"), "{}", l.hover);
        assert_eq!(l.forget, Some(id));
        let t = trust(Some(&m), &scene, &st);
        assert_eq!(t.text, format!("no — {}", l.text));
        assert_eq!(t.forget, Some(id));
        // without a location: today's line, no link
        let m = ModelStatus { trust_checks: checks(old_f.clone()), ..m };
        let t = trust(Some(&m), &scene, &st);
        assert_eq!(t.text, "no — Reproduces its own training forces — worst 3.4, allowed 2 mHa/Bohr for that atom");
        assert_eq!(t.forget, None);
        assert!(activity::trust::lines(&m, &scene, &st).iter().all(|l| l.forget.is_none()));
    }

    /// Plan review M3: the class line words the class's own verdict, so an energy-only failure
    /// is an energy error.
    #[test]
    fn the_class_line_words_the_failing_condition() {
        use activity::words::class_line;
        use sim::{ClassKey, ClassLoo};
        let l = |class, refusal: Option<(ClassKey, f64, f64)>| ClassLoo { class, samples: 120, force_rms: 0.1, force_max: 0.4, energy_rms: 300.0, energy_max: 1100.0, trusted: refusal.is_none(), refusal, calibration: 2.1, measured: Some(1.4), measured_energy: Some(0.2), worst_atom: Some(2.2), evidence: 12, trusted_since: None, inflation: 1.0, species: Vec::new() };
        assert_eq!(class_line(&l("pair", None)), "Two-atom arrangements — learned");
        assert_eq!(class_line(&l("pair", Some((ClassKey::Energy, 1100.0, 1.0)))), "Two-atom arrangements — not yet (leave-one-out energy error 1100.0× the tier)");
        assert_eq!(class_line(&l("pair", Some((ClassKey::Force, 2.69, 1.0)))), "Two-atom arrangements — not yet (leave-one-out force error 2.7× the bound)");
        assert_eq!(class_line(&l("many-body", Some((ClassKey::Samples, 12.0, 40.0)))), "Three-or-more-atom arrangements — not yet (12 of 40 samples needed)");
        // M8a Task 2 (fix round 2, review I3): each number named by what measured it — the DFT
        // checks' measured error, the fit's leave-one-out floor, the measured energy
        assert_eq!(class_line(&l("pair", Some((ClassKey::Calibrated, 1.4, 1.0)))), "Two-atom arrangements — not yet (DFT checks measured its force error at 1.40× the bound over 12 DFT checks, limit 1.00×)");
        assert_eq!(class_line(&l("pair", Some((ClassKey::LooFloor, 0.8, std::f64::consts::FRAC_1_SQRT_2)))), "Two-atom arrangements — not yet (leave-one-out force error 0.80× the bound on average, limit 0.71×)");
        assert_eq!(class_line(&l("pair", Some((ClassKey::MeasuredEnergy, 0.9, std::f64::consts::FRAC_1_SQRT_2)))), "Two-atom arrangements — not yet (DFT checks measured its energy error at 0.90× the tier over 12 DFT checks, limit 0.71×)");
        // open-shell model (the energy half of C1): the energy RMS with the misfits counted
        assert_eq!(class_line(&l("pair", Some((ClassKey::EnergyRms, 1.3, 1.0)))), "Two-atom arrangements — not yet (leave-one-out energy error 1.3× the tier on average, training samples the model cannot reproduce included)");
        // the worst-atom guard: on entry with evidence, and both ways with fewer than three checks
        assert_eq!(class_line(&l("pair", Some((ClassKey::WorstAtom, 4.2, 3.0)))), "Two-atom arrangements — not yet (a recent DFT check found an atom's force 4.2× its bound; it starts answering only without one beyond 3×)");
        assert_eq!(class_line(&ClassLoo { evidence: 2, ..l("pair", Some((ClassKey::WorstAtom, 4.2, 3.0))) }), "Two-atom arrangements — not yet (a DFT check found an atom's force 4.2× its bound, with fewer than 3 checks to judge the class by)");
        // no claim of the posterior's calibration is worded (no "1.0× less", no "inf× less")
        for k in [ClassKey::Calibrated, ClassKey::LooFloor, ClassKey::MeasuredEnergy, ClassKey::WorstAtom, ClassKey::Window] {
            for cal in [1.0, f64::INFINITY] {
                let t = class_line(&ClassLoo { calibration: cal, ..l("pair", Some((k, 1.4, 1.0))) });
                assert!(!t.contains("less") && !t.contains("inf"), "{t}");
            }
        }
    }

    /// M8a Task 2 (the Task 1 physics review's I4): audits near the cell faces that could not
    /// judge the model are said on the checklist's audit line, never worded as a pass.
    #[test]
    fn box_affected_audits_are_said_on_the_audit_line() {
        use activity::trust::lines;
        let scene = Scene::preset("h2").unwrap();
        let st = UiState::new("h2", &scene);
        let mut m = model();
        m.trust_checks = vec![sim::activity::TrustCheck { key: sim::activity::TrustKey::Audit, met: true, value: 0.0, limit: 1e-3, worst: None }];
        let plain = lines(&m, &scene, &st);
        assert!(!plain.iter().any(|l| l.text.contains("near the cell faces")), "{plain:?}");
        m.audits_box_unjudged = 3;
        m.last_audit.as_mut().unwrap().verdict = sim::AuditVerdict::BoxAffected;
        let said = lines(&m, &scene, &st);
        let line = said.iter().find(|l| l.text.starts_with("Last check against DFT")).expect("the audit line");
        assert!(!line.met && !line.why, "unverified: no tick, and not the refusal's reason either (review M5)");
        assert!(line.text.contains("box-affected, not judged") && line.text.ends_with("— unverified: 3 audits near the cell faces could not judge the model"), "{}", line.text);
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
                TrustCheck { key: TrustKey::Samples, met: true, value: 145.0, limit: 29.0, worst: None },
                TrustCheck { key: TrustKey::ResidualDofE, met: true, value: 120.0, limit: 30.0, worst: None },
                TrustCheck { key: TrustKey::ResidualDofF, met: true, value: 700.0, limit: 30.0, worst: None },
                TrustCheck { key: TrustKey::EnergyResidual, met: false, value: 2.2e-3, limit: 1e-3, worst: None },
                TrustCheck { key: TrustKey::ForceResidual, met: true, value: 1e-3, limit: 2e-3, worst: None },
                TrustCheck { key: TrustKey::ForceNoise, met: true, value: 4e-4, limit: 2e-3, worst: None },
                TrustCheck { key: TrustKey::Audit, met: true, value: 0.0, limit: 1e-3, worst: None },
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
        // the misfit row too (spec 2026-11-13 §4.3): before it, those samples failed the residual rows
        for key in [K::EnergyResidual, K::ForceResidual, K::Misfit, K::ForceNoise] {
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
            trust_checks: vec![TrustCheck { key: TrustKey::Samples, met: false, value: 12.0, limit: 16.0, worst: None }],
            ..model()
        };
        let scene = Scene::preset("h2").unwrap();
        let st = UiState::new("h2", &scene);
        let lines = activity::trust::lines(&m, &scene, &st);
        assert!(lines[0].why && lines[0].text == "Enough samples in the fit — 12 of 16 needed (one per independent direction the model can bend)");
        assert_eq!(trust(Some(&m), &scene, &st).text, "no — Enough samples in the fit — 12 of 16 needed (one per independent direction the model can bend)");
        // met: every line ✓, none "this is why", the box line ✓, the class lines after
        let ok = ModelStatus { trusted: true, untrusted_reason: None, trust_checks: vec![TrustCheck { key: TrustKey::Samples, met: true, value: 145.0, limit: 16.0, worst: None }], ..model() };
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
            trust_checks: vec![TrustCheck { key: TrustKey::Samples, met: true, value: 145.0, limit: 16.0, worst: None }],
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
    use providers::pseudo::GenerationCost as G;
    let gen = PanelInput { pseudo_generating: Some((1, 12.4, G::GpuGate)), ..input(&scene, None, &cam) };
    let notes = overlays::chip_notes(&st, &gen);
    assert!(notes.iter().any(|(_, t)| t == "generating the H pseudopotential (≈ 1 min: its dimer check)… 12 s"), "{notes:?}");
    // final review M6: the CPU gate's 40 minutes are said
    let cpu = PanelInput { pseudo_generating: Some((1, 75.0, G::CpuGate)), ..input(&scene, None, &cam) };
    let notes = overlays::chip_notes(&st, &cpu);
    assert!(notes.iter().any(|(_, t)| t == "generating the H pseudopotential (≈ 40 min on the CPU: its dimer check (no GPU))… 75 s"), "{notes:?}");
    let o = PanelInput { pseudo_generating: Some((8, 3.0, G::Seconds)), ..input(&scene, None, &cam) };
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
        Scene { nuclei, charge: 0, multiplicity: None, grid_n: 2 * (6.0 / h).round() as usize + 1, grid_h: h, sigma: None, ionic: sim::IonicPolicy::CoresOnly, train_mask: None }
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

/// Docking spec D-2..D-4: one workspace tree. The default (spec D-9) shows Model, Selection, View
/// and Log, each the active tab of its leaf; clicking a tab shows it and hides only the other tabs
/// of its leaf; `open_tab` shows a panel where it is, and brings a closed one back beside the 3D
/// view on the right; Ctrl K reveals a control in its own panel; every arrangement draws without
/// a command.
#[test]
fn the_workspace_shows_one_tab_per_leaf() {
    use super::workspace::layout;
    use DockTab::*;
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    let shown = |st: &UiState| DockTab::PANELS.into_iter().filter(|&t| st.tab_shown(t)).collect::<Vec<_>>();
    assert_eq!(shown(&st), vec![Model, Selection, View, Log], "the D1 default");
    assert_eq!(layout::encode(&st.workspace.to_node()), layout::encode(&layout::default_layout()));
    for _ in 0..2 {
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
    }
    // a tab click shows its panel, in its own leaf only (Activity and Graph are tabbed with the
    // Log since docking Task 4)
    click_widget(&ctx, &mut st, &inp, "dock-tab-Graph");
    assert_eq!(shown(&st), vec![Model, Selection, View, Graph]);
    click_widget(&ctx, &mut st, &inp, "dock-tab-Knowledge");
    assert_eq!(shown(&st), vec![Knowledge, Selection, View, Graph]);
    click_widget(&ctx, &mut st, &inp, "dock-tab-Orbitals");
    assert_eq!(shown(&st), vec![Knowledge, Selection, Orbitals, Graph]);
    // open_tab shows a panel where it is
    st.open_tab(Readouts);
    assert_eq!(shown(&st), vec![Knowledge, Readouts, Orbitals, Graph]);
    st.open_tab(Log);
    assert_eq!(shown(&st), vec![Knowledge, Readouts, Orbitals, Log]);
    // closed, it is gone; open_tab brings it back in the right column's first leaf
    st.workspace.close_panel(Graph);
    assert!(!st.tab_shown(Graph) && !layout::panels(&st.workspace.to_node()).contains(&Graph));
    st.open_tab(Graph);
    assert!(st.tab_shown(Graph));
    assert_eq!(layout::encode(&st.workspace.to_node()), "H(0.24,V(0.55,[Model,Knowledge*],[Selection,Readouts*]),H(0.70,3D,V(0.62,[View,Orbitals,Graph*],[Log*,Activity])))");
    // the right column closed: a panel shown again gets a new split right of the view
    for t in [View, Orbitals, Graph, Log, Activity] {
        st.workspace.close_panel(t);
    }
    st.open_tab(Log);
    assert_eq!(layout::encode(&st.workspace.to_node()), "H(0.24,V(0.55,[Model,Knowledge*],[Selection,Readouts*]),H(0.70,3D,[Log*]))");
    st.reset_layout();
    assert_eq!(shown(&st), vec![Model, Selection, View, Log], "Reset layout: the default back");
    // Ctrl K reveals a control in its own panel: the solve log (Activity), then the log itself
    // (the same leaf: Activity goes behind it)
    let solves = search::results("solve log", 1)[0];
    assert_eq!(solves.place, search::Place::Dock(Activity));
    search::pick(&ctx, &mut st, solves);
    assert!(st.tab_shown(Activity) && !st.tab_shown(Log));
    let log = search::ENTRIES.iter().find(|e| e.anchor == "d-log").unwrap();
    search::pick(&ctx, &mut st, log);
    assert!(st.tab_shown(Log) && !st.tab_shown(Activity));
    // and a View-panel control shows the View panel
    st.open_tab(Orbitals);
    search::pick(&ctx, &mut st, search::results("bond angles", 1)[0]);
    assert!(st.tab_shown(View) && !st.tab_shown(Orbitals));
    // the chrome renders every arrangement without a command or an error: the default, every
    // panel in one leaf, the 3D view between stacked leaves, every panel closed
    for text in [None, Some("H(0.4,[Model,Knowledge,Selection,Readouts,Activity,Graph,View,Orbitals,Log*],3D)"), Some("V(0.3,[Log*],V(0.5,3D,[Graph*,Readouts]))")] {
        match text {
            Some(t) => st.workspace.set_node(layout::parse(t).unwrap()),
            None => st.reset_layout(),
        }
        assert!(layout::panels(&st.workspace.to_node()).len() == 9 || text.is_some());
        for w in [1400.0, 1000.0] {
            assert!(frame(&ctx, &mut st, &inp, w, vec![]).commands.is_empty(), "{text:?} at {w}");
        }
    }
    for t in DockTab::PANELS {
        st.workspace.close_panel(t);
    }
    assert!(shown(&st).is_empty());
    for w in [1400.0, 1000.0] {
        let out = frame(&ctx, &mut st, &inp, w, vec![]);
        assert!(out.commands.is_empty(), "every panel closed at {w}");
    }
}

/// Docking spec D-3: the 3D view is the tile between the columns — its rect is
/// `PanelOutput::view`, non-empty, right of the left column's tabs and left of the right
/// column's — and the pointer over it is not over egui (the mouse tools get it, as over the
/// space the old docks left), while over a panel it is. Wherever the tile sits.
#[test]
fn the_3d_view_is_the_tile_between_the_columns() {
    use super::workspace::layout;
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    let mut view = egui::Rect::NOTHING;
    for _ in 0..2 {
        view = frame(&ctx, &mut st, &inp, 1400.0, vec![]).view.unwrap();
    }
    let probe = |name: &str| PROBES.with(|p| p.borrow()[name]);
    let (model, log) = (probe("dock-tab-Model"), probe("dock-tab-Log"));
    assert!(view.width() > 500.0 && view.height() > 500.0, "{view:?}");
    assert!(model.right() <= view.left() && view.right() <= log.left(), "model tab {model:?}, view {view:?}, log tab {log:?}");
    assert!((330.0..350.0).contains(&view.left()) && (1060.0..1100.0).contains(&view.right()), "24 % of 1400, then 70 % of the rest: {view:?}");
    let over = |st: &mut UiState, at: egui::Pos2| {
        frame(&ctx, st, &inp, 1400.0, vec![egui::Event::PointerMoved(at)]);
        (ctx.is_pointer_over_egui(), ctx.egui_wants_pointer_input())
    };
    assert_eq!(over(&mut st, view.center()), (false, false), "the 3D view is not egui's");
    assert_eq!(over(&mut st, model.center() + egui::vec2(0.0, 120.0)), (true, true), "the Model panel's body is");
    assert_eq!(over(&mut st, log.center()), (true, true), "a tab is");
    // the view's tile in another place: under a panel, at the window's right
    st.workspace.set_node(layout::parse("H(0.3,[Model*],V(0.4,[Log*],3D))").unwrap());
    for _ in 0..2 {
        view = frame(&ctx, &mut st, &inp, 1400.0, vec![]).view.unwrap();
    }
    let (model, log) = (probe("dock-tab-Model"), probe("dock-tab-Log"));
    assert!(model.right() <= view.left() && log.bottom() <= view.top() && view.width() > 500.0, "{view:?}");
    assert_eq!(over(&mut st, view.center()), (false, false), "{view:?}");
    assert_eq!(over(&mut st, log.center() + egui::vec2(0.0, 80.0)), (true, true), "the Log panel above it");
}

/// The workspace is display state: clicking its tabs, closing and showing panels records no
/// action and sends no command.
#[test]
fn the_workspace_is_not_recorded() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    let mut outs = vec![click_widget(&ctx, &mut st, &inp, "dock-tab-Graph"), click_widget(&ctx, &mut st, &inp, "dock-tab-Log")];
    st.workspace.close_panel(DockTab::Log);
    outs.push(vec![frame(&ctx, &mut st, &inp, 1400.0, vec![])]);
    st.open_tab(DockTab::Log);
    outs.push(click_widget(&ctx, &mut st, &inp, "dock-tab-Knowledge"));
    assert!(outs.iter().flatten().all(|o| o.actions.is_empty() && o.commands.is_empty()), "no action, no command");
}

/// Docking spec D-3: `egui_dock` cannot refuse a drop into the 3D view's leaf, so a panel found
/// there is moved to a new split right of the view on the same frame (the pass is redrawn), still
/// active; the view's leaf keeps only the view.
#[test]
fn a_panel_dropped_onto_the_3d_view_goes_beside_it() {
    use super::workspace::layout;
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    st.workspace.set_node(layout::parse("H(0.3,[Model*],3D)").unwrap());
    st.workspace.drop_into_view_for_test(DockTab::Log);
    let out = frame(&ctx, &mut st, &inp, 1400.0, vec![]);
    assert_eq!(layout::encode(&st.workspace.to_node()), "H(0.30,[Model*],H(0.70,3D,[Log*]))");
    assert!(st.tab_shown(DockTab::Log));
    let view = out.view.unwrap();
    // the pass with the Log over the whole 70 % (≈ 980 pt) is discarded: the frame hands back the
    // redrawn one, the view in 70 % of that (≈ 686 pt), the same rect the next frame draws
    let next = frame(&ctx, &mut st, &inp, 1400.0, vec![]).view.unwrap();
    assert_eq!(view, next, "the drop frame's view is the redrawn pass's");
    assert!((500.0..700.0).contains(&view.width()), "the view beside the Log, not under it: {view:?}");
}

/// `egui_dock` clamps each split's fraction every frame to keep `separator.extra` on both sides,
/// and the profile saves what it left: a narrow window must not rewrite the default layout.
#[test]
fn a_narrow_window_keeps_the_saved_fractions() {
    use super::workspace::layout;
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    let before = layout::encode(&st.workspace.to_node());
    for _ in 0..3 {
        frame(&ctx, &mut st, &inp, 700.0, vec![]);
    }
    assert_eq!(layout::encode(&st.workspace.to_node()), before, "at 700 pt wide");
}

/// A tiny window (320 × 240, and 1 × 1) draws the whole chrome with the default layout for a few
/// frames without a panic; the 3D view's rect is there and finite.
#[test]
fn a_tiny_window_draws() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    for size in [egui::vec2(320.0, 240.0), egui::vec2(1.0, 1.0), egui::vec2(320.0, 240.0)] {
        for _ in 0..3 {
            let out = frame_sized(&ctx, &mut st, &inp, size, vec![]);
            let view = out.view.unwrap_or_else(|| panic!("no view at {size:?}"));
            assert!(view.min.is_finite() && view.max.is_finite(), "{view:?} at {size:?}");
            assert!(out.commands.is_empty(), "at {size:?}");
        }
    }
    // with panels auto-hidden on every edge and a flyout out (docking spec D-10)
    use super::workspace::layout::{Edge, Hidden};
    st.reset_layout();
    st.workspace.set_hidden(vec![]);
    for (t, edge) in [(DockTab::Graph, Edge::Left), (DockTab::Log, Edge::Right), (DockTab::Readouts, Edge::Bottom)] {
        st.workspace.close_panel(t);
        st.workspace.pin(t, false);
        let h = st.workspace.hidden().iter().position(|h| h.tab == t).unwrap();
        let mut all = st.workspace.hidden().to_vec();
        all[h] = Hidden { tab: t, edge, size: 400.0 };
        st.workspace.set_hidden(all);
    }
    assert_eq!(st.workspace.hidden().len(), 3);
    for t in [DockTab::Graph, DockTab::Log, DockTab::Readouts] {
        st.open_tab(t);
        for size in [egui::vec2(320.0, 240.0), egui::vec2(1.0, 1.0), egui::vec2(320.0, 240.0)] {
            for _ in 0..3 {
                let out = frame_sized(&ctx, &mut st, &inp, size, vec![]);
                let view = out.view.unwrap_or_else(|| panic!("no view at {size:?}"));
                assert!(view.min.is_finite() && view.max.is_finite(), "{view:?} at {size:?} with {t:?} out");
                assert!(out.commands.is_empty(), "at {size:?}");
            }
        }
        assert_eq!(st.workspace.flyout(), Some(t), "still out (opened by open_tab)");
    }
}

/// Docking spec D-4: a tab that can carry a status dot keeps the dot's room whether the dot is
/// there or not, so the tab bar does not jump when work starts or stops (or the Log's count
/// comes and goes).
#[test]
fn a_tab_keeps_its_width_when_its_dot_comes_and_goes() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    // the Log behind another tab (on screen, it reads its count away)
    st.workspace.set_node(super::workspace::layout::parse("H(0.70,3D,[Graph*,Log,View])").unwrap());
    let probe = |name: &str| PROBES.with(|p| p.borrow()[name]);
    for _ in 0..2 {
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
    }
    let (quiet, view) = (probe("dock-tab-Log"), probe("dock-tab-View"));
    st.log.unread = 3;
    for _ in 0..2 {
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
    }
    assert!(dock::status(DockTab::Log, &st, &inp).is_some(), "the Log's count shows");
    assert_eq!(probe("dock-tab-Log"), quiet, "the Log's tab with and without its count");
    assert_eq!(probe("dock-tab-View"), view, "the tab after it holds still");
    assert_eq!(dock::icon(DockTab::View), super::icons::Icon::View, "the View panel has its own icon");
}

/// A leaf's close button paints the duotone Close icon over `egui_dock`'s close-all button:
/// clicking where it is painted closes the leaf's panels.
#[test]
fn the_painted_close_button_closes_its_leaf() {
    use super::workspace::layout;
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    st.workspace.set_node(layout::parse("H(0.70,3D,V(0.50,[Graph*,Orbitals],[Log*]))").unwrap());
    for _ in 0..2 {
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
    }
    let close = probed_rect("dock-close-Graph");
    let graph = probed_rect("dock-tab-Graph");
    assert!(close.left() > graph.right() && (close.center().y - graph.center().y).abs() < 4.0, "close {close:?} at the bar's end, tab {graph:?}");
    let outs = click_widget(&ctx, &mut st, &inp, "dock-close-Graph");
    assert!(outs.iter().all(|o| o.actions.is_empty() && o.commands.is_empty()));
    assert_eq!(layout::encode(&st.workspace.to_node()), "H(0.70,3D,[Log*])");
}

/// The profile keeps the workspace as one `layout = …` line (docking spec D-8) and no old dock
/// keys; it reads back to the same tree; a layout that does not parse loads the default.
#[test]
fn the_profile_keeps_the_layout() {
    use super::workspace::layout;
    let scene = Scene::preset("h2").unwrap();
    let mut st = UiState::new("h2", &scene);
    let text = profile::capture(&st);
    assert!(text.contains(&format!("layout = {}\n", layout::encode(&layout::default_layout()))), "{text}");
    for old in ["left_tab", "right_tab", "left_width", "right_width", "dock_open", "left_open"] {
        assert!(!text.contains(old), "{old} in\n{text}");
    }
    assert_eq!(profile::defaults(&st), text, "a fresh state's profile is the defaults");
    let custom = "V(0.30,[Log*,Graph],H(0.60,3D,[Model,View*]))";
    st.workspace.set_node(layout::parse(custom).unwrap());
    let text = profile::capture(&st);
    let mut back = UiState::new("h2", &scene);
    profile::apply(&mut back, &text);
    assert_eq!(layout::encode(&back.workspace.to_node()), custom);
    assert!(back.tab_shown(DockTab::Log) && back.tab_shown(DockTab::View) && !back.tab_shown(DockTab::Model));
    // nonsense, or a tree without the 3D view: the default
    for bad in ["layout = nonsense", "layout = [Model*,Log]", "layout = "] {
        let mut s = UiState::new("h2", &scene);
        s.workspace.set_node(layout::parse(custom).unwrap());
        profile::apply(&mut s, bad);
        assert_eq!(s.workspace.to_node(), layout::default_layout(), "{bad}");
    }
    // a profile without layout keys leaves the arrangement alone
    let mut s = UiState::new("h2", &scene);
    s.workspace.set_node(layout::parse(custom).unwrap());
    profile::apply(&mut s, "show_bonds = false\n");
    assert_eq!(layout::encode(&s.workspace.to_node()), custom);
}

// ---- auto-hide (docking spec D-10) ----

/// One 1400 × 900 frame at input time `time` (s): the auto-hide timers run on egui's input time.
fn frame_at(ctx: &egui::Context, st: &mut UiState, inp: &PanelInput<'_>, time: f64, events: Vec<egui::Event>) -> PanelOutput {
    let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0))), time: Some(time), events, ..Default::default() };
    let mut out = None;
    let mut full = ctx.run_ui(raw, |root| out = Some(show(root, st, inp)));
    full.textures_delta.clear();
    out.unwrap()
}

/// Spec D-10: unpinning takes a panel out of the tree onto the strip of the 3D view's edge its
/// leaf lay beyond, remembering its size (the left column's Readouts: left, its 24 % width; the
/// right column's Graph: right; a leaf under the view: bottom, its height); the strips are carved
/// from the 3D view; `open_tab` opens its flyout at that edge (`tab_shown` while it is out); its
/// pin docks it back beside the 3D view on that edge.
#[test]
fn an_unpinned_panel_folds_to_its_edge_and_pins_back() {
    use super::workspace::layout::{self, Edge};
    use DockTab::*;
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    let mut tile = egui::Rect::NOTHING;
    for _ in 0..2 {
        tile = frame(&ctx, &mut st, &inp, 1400.0, vec![]).view.unwrap();
    }
    st.open_tab(Readouts);
    st.workspace.pin(Readouts, false);
    assert!(!layout::panels(&st.workspace.to_node()).contains(&Readouts) && !st.tab_shown(Readouts), "out of the tree");
    let h = st.workspace.hidden()[0];
    assert_eq!((h.tab, h.edge), (Readouts, Edge::Left));
    assert!((320.0..345.0).contains(&h.size), "the left column's width, 24 % of 1400: {}", h.size);
    st.workspace.pin(Graph, false);
    assert_eq!(st.workspace.hidden()[1].edge, Edge::Right);
    assert_eq!(layout::encode(&st.workspace.to_node()), "H(0.24,V(0.55,[Model*,Knowledge],[Selection*]),H(0.70,3D,V(0.62,[View*,Orbitals],[Log*,Activity])))");
    let mut view = egui::Rect::NOTHING;
    for _ in 0..2 {
        view = frame(&ctx, &mut st, &inp, 1400.0, vec![]).view.unwrap();
    }
    let (left, right) = (probed_rect("strip-Left"), probed_rect("strip-Right"));
    assert!(left.contains_rect(probed_rect("strip-icon-Readouts")) && right.contains_rect(probed_rect("strip-icon-Graph")), "each on its edge's strip");
    assert!((left.left() - tile.left()).abs() < 1.0 && (left.right() - view.left()).abs() < 1.0, "the left strip at the view's left: {left:?}, view {view:?}");
    assert!((right.right() - tile.right()).abs() < 1.0 && (right.left() - view.right()).abs() < 1.0, "{right:?}, view {view:?}");
    assert!((tile.width() - view.width() - 2.0 * super::workspace::autohide::STRIP).abs() < 1.0, "the strips are carved from the tile: {tile:?} → {view:?}");
    // open_tab: its flyout, against the left strip at its remembered width
    st.open_tab(Readouts);
    assert!(st.tab_shown(Readouts) && st.workspace.flyout() == Some(Readouts));
    for _ in 0..12 {
        frame(&ctx, &mut st, &inp, 1400.0, vec![]); // past the slide
    }
    let fly = probed_rect("flyout-Readouts");
    assert!((fly.left() - view.left()).abs() < 1.0 && (fly.width() - h.size).abs() < 1.0 && (fly.height() - view.height()).abs() < 1.0, "{fly:?} in {view:?}");
    // its pin: back beside the 3D view on the left
    let outs = click_widget(&ctx, &mut st, &inp, "flyout-pin-Readouts");
    assert!(outs.iter().all(|o| o.actions.is_empty() && o.commands.is_empty()), "display state");
    assert_eq!(st.workspace.flyout(), None);
    assert_eq!(st.workspace.hidden().iter().map(|h| h.tab).collect::<Vec<_>>(), vec![Graph]);
    let enc = layout::encode(&st.workspace.to_node());
    assert!(enc.starts_with("H(0.24,V(0.55,[Model*,Knowledge],[Selection*]),H(0.70,H(") && enc.contains(",[Readouts*],3D),"), "{enc}");
    assert!(st.tab_shown(Readouts));
    // a leaf under the 3D view folds to the bottom at its height, and docks back there
    st.workspace.set_node(layout::parse("H(0.70,V(0.60,3D,[Log*]),[Model*])").unwrap());
    assert!(st.workspace.hidden().is_empty(), "a new layout: nothing hidden");
    for _ in 0..2 {
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
    }
    st.workspace.pin(Log, false);
    let h = st.workspace.hidden()[0];
    assert_eq!((h.tab, h.edge), (Log, Edge::Bottom));
    assert!((250.0..360.0).contains(&h.size), "40 % of the column's height: {}", h.size);
    for _ in 0..2 {
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
    }
    let bottom = probed_rect("strip-Bottom");
    assert!(bottom.width() > bottom.height() && bottom.contains_rect(probed_rect("strip-icon-Log")), "the bottom strip runs along the view");
    st.workspace.pin(Log, true);
    assert_eq!(layout::encode(&st.workspace.to_node()), "H(0.70,V(0.60,3D,[Log*]),[Model*])", "back where it was");
    // a closed panel unpins to the right edge; pinning a closed panel shows it
    st.workspace.close_panel(Graph);
    st.workspace.pin(Graph, false);
    assert_eq!(st.workspace.hidden()[0].edge, Edge::Right);
    st.workspace.close_panel(Graph);
    assert!(st.workspace.hidden().is_empty() && !st.tab_shown(Graph), "closing a hidden panel takes it off its strip");
    st.workspace.pin(Graph, true);
    assert!(st.tab_shown(Graph) && layout::panels(&st.workspace.to_node()).contains(&Graph));
    st.reset_layout();
    assert!(st.workspace.hidden().is_empty());
}

/// Unpinned before the tree is first drawn (a layout just set, a preset): the edge is the side of
/// the 3D view the panel sits on in the tree, at the default size.
#[test]
fn a_panel_unpinned_before_drawing_folds_to_its_side_in_the_tree() {
    use super::workspace::layout::{self, Edge};
    use DockTab::*;
    let scene = Scene::preset("h2").unwrap();
    let mut st = UiState::new("h2", &scene);
    for t in [Readouts, Model, Graph, View] {
        st.workspace.pin(t, false);
    }
    let edges: Vec<_> = st.workspace.hidden().iter().map(|h| (h.tab, h.edge, h.size)).collect();
    let d = super::workspace::autohide::DEFAULT_SIZE;
    assert_eq!(edges, vec![(Readouts, Edge::Left, d), (Model, Edge::Left, d), (Graph, Edge::Right, d), (View, Edge::Right, d)]);
    st.workspace.set_node(layout::parse("V(0.30,[Model*],V(0.60,3D,[Log*]))").unwrap());
    st.workspace.pin(Log, false);
    st.workspace.pin(Model, false);
    let edges: Vec<_> = st.workspace.hidden().iter().map(|h| (h.tab, h.edge)).collect();
    assert_eq!(edges, vec![(Log, Edge::Bottom), (Model, Edge::Right)], "below the view: bottom; above it (no top strip): right");
}

/// Spec D-10's timings, on egui's input time: hovering a strip icon for 300 ms (frames at 60 Hz)
/// opens its flyout — not before 250 ms — and leaving the flyout and the strip for 600 ms hides
/// it — not at 400 ms; over the flyout it stays.
#[test]
fn hovering_a_strip_icon_opens_its_flyout_and_leaving_hides_it() {
    use super::workspace::layout;
    use DockTab::Graph;
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    st.workspace.set_node(layout::parse("H(0.30,[Graph*],3D)").unwrap());
    let dt = 1.0 / 60.0;
    let mut t = 10.0;
    let mut step = |st: &mut UiState, events: Vec<egui::Event>| {
        let out = frame_at(&ctx, st, &inp, t, events);
        t += dt;
        out
    };
    for _ in 0..2 {
        step(&mut st, vec![]);
    }
    st.workspace.pin(Graph, false);
    let mut view = egui::Rect::NOTHING;
    for _ in 0..2 {
        view = step(&mut st, vec![]).view.unwrap();
    }
    let icon = probed_rect("strip-icon-Graph").center();
    for i in 0..=18 {
        step(&mut st, vec![egui::Event::PointerMoved(icon)]);
        if i <= 13 {
            assert_eq!(st.workspace.flyout(), None, "{} ms of hover", i * 1000 / 60);
        }
    }
    assert_eq!(st.workspace.flyout(), Some(Graph), "open after 300 ms of hover");
    assert!(st.tab_shown(Graph));
    // over the flyout it stays out
    for _ in 0..12 {
        step(&mut st, vec![]);
    }
    let fly = probed_rect("flyout-Graph");
    for _ in 0..40 {
        step(&mut st, vec![egui::Event::PointerMoved(fly.center())]);
    }
    assert_eq!(st.workspace.flyout(), Some(Graph), "the pointer on the flyout");
    // away from both: still out at 400 ms, gone by 600 ms
    let away = egui::pos2(view.right() - 40.0, view.center().y);
    assert!(!fly.contains(away));
    for i in 0..=36 {
        step(&mut st, vec![egui::Event::PointerMoved(away)]);
        if i <= 24 {
            assert_eq!(st.workspace.flyout(), Some(Graph), "{} ms away", i * 1000 / 60);
        }
    }
    assert_eq!(st.workspace.flyout(), None, "hidden after 600 ms away");
    assert!(!st.tab_shown(Graph));
}

/// Spec D-10: a click on a strip icon keeps the flyout out until the next click elsewhere (the
/// pointer away for a second does not hide it); Esc hides it — and does not also deselect
/// (`keys`); so does a click elsewhere after `open_tab`. A press inside the flyout keeps a
/// hover-opened flyout out too (working in it). Nothing is recorded.
#[test]
fn a_clicked_flyout_stays_until_a_click_elsewhere_or_esc() {
    use super::workspace::layout;
    use DockTab::Graph;
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    st.workspace.set_node(layout::parse("H(0.30,[Graph*],3D)").unwrap());
    let mut outs = Vec::new();
    let mut t = 10.0;
    macro_rules! step {
        ($st:expr, $events:expr) => {{
            outs.push(frame_at(&ctx, $st, &inp, t, $events));
            t += 1.0 / 60.0;
        }};
    }
    for _ in 0..2 {
        step!(&mut st, vec![]);
    }
    st.workspace.pin(Graph, false);
    for _ in 0..2 {
        step!(&mut st, vec![]);
    }
    let icon = probed_rect("strip-icon-Graph").center();
    step!(&mut st, vec![egui::Event::PointerMoved(icon), pointer(icon, true)]);
    step!(&mut st, vec![pointer(icon, false)]);
    assert_eq!(st.workspace.flyout(), Some(Graph), "a click opens it at once");
    for _ in 0..12 {
        step!(&mut st, vec![]);
    }
    let fly = probed_rect("flyout-Graph");
    let away = egui::pos2(fly.right() + 200.0, fly.center().y);
    for _ in 0..60 {
        step!(&mut st, vec![egui::Event::PointerMoved(away)]);
    }
    assert_eq!(st.workspace.flyout(), Some(Graph), "clicked open: a second away does not hide it");
    // a click inside it does not hide it either
    step!(&mut st, vec![egui::Event::PointerMoved(fly.center()), pointer(fly.center(), true)]);
    step!(&mut st, vec![pointer(fly.center(), false)]);
    assert_eq!(st.workspace.flyout(), Some(Graph));
    // a click elsewhere does
    step!(&mut st, vec![egui::Event::PointerMoved(away), pointer(away, true)]);
    step!(&mut st, vec![pointer(away, false)]);
    assert_eq!(st.workspace.flyout(), None, "a click on the 3D view hides it");
    // open_tab opens it the same way; Esc hides it and does not deselect
    st.open_tab(Graph);
    for _ in 0..60 {
        step!(&mut st, vec![egui::Event::PointerMoved(away)]);
    }
    assert_eq!(st.workspace.flyout(), Some(Graph), "open_tab: stuck open");
    step!(&mut st, vec![key(egui::Key::Escape, egui::Modifiers::NONE)]);
    assert_eq!(st.workspace.flyout(), None, "Esc hides it");
    // hover-opened, then worked in: it stays when the pointer leaves
    for _ in 0..20 {
        step!(&mut st, vec![egui::Event::PointerMoved(icon)]);
    }
    assert_eq!(st.workspace.flyout(), Some(Graph));
    step!(&mut st, vec![egui::Event::PointerMoved(fly.center()), pointer(fly.center(), true)]);
    step!(&mut st, vec![pointer(fly.center(), false)]);
    for _ in 0..60 {
        step!(&mut st, vec![egui::Event::PointerMoved(away)]);
    }
    assert_eq!(st.workspace.flyout(), Some(Graph), "a press inside keeps it out");
    step!(&mut st, vec![key(egui::Key::Escape, egui::Modifiers::NONE)]);
    assert!(outs.iter().all(|o| o.actions.is_empty() && o.commands.is_empty() && !o.deselect), "display state; Esc on a flyout does not deselect");
    // with no flyout, Esc deselects as before
    step!(&mut st, vec![key(egui::Key::Escape, egui::Modifiers::NONE)]);
    let _ = t;
    assert!(outs.last().unwrap().actions.contains(&Action::Deselect));
}

/// Esc while the pointer rests on the strip icon hides the flyout and it stays hidden — the hover
/// timer starts again only once the pointer has left the icon and come back.
#[test]
fn esc_with_the_pointer_on_the_strip_icon_keeps_the_flyout_hidden() {
    use super::workspace::layout;
    use DockTab::Graph;
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    st.workspace.set_node(layout::parse("H(0.30,[Graph*],3D)").unwrap());
    let mut t = 10.0;
    let mut step = |st: &mut UiState, events: Vec<egui::Event>| {
        frame_at(&ctx, st, &inp, t, events);
        t += 1.0 / 60.0;
    };
    for _ in 0..2 {
        step(&mut st, vec![]);
    }
    st.workspace.pin(Graph, false);
    for _ in 0..2 {
        step(&mut st, vec![]);
    }
    let icon = probed_rect("strip-icon-Graph").center();
    for _ in 0..20 {
        step(&mut st, vec![egui::Event::PointerMoved(icon)]);
    }
    assert_eq!(st.workspace.flyout(), Some(Graph), "hover-opened");
    step(&mut st, vec![key(egui::Key::Escape, egui::Modifiers::NONE)]);
    assert_eq!(st.workspace.flyout(), None, "Esc");
    for _ in 0..40 {
        step(&mut st, vec![egui::Event::PointerMoved(icon)]);
    }
    assert_eq!(st.workspace.flyout(), None, "resting on the icon does not reopen it");
    let view = probed_rect("strip-Left").right() + 300.0;
    step(&mut st, vec![egui::Event::PointerMoved(egui::pos2(view, icon.y))]);
    for _ in 0..20 {
        step(&mut st, vec![egui::Event::PointerMoved(icon)]);
    }
    assert_eq!(st.workspace.flyout(), Some(Graph), "back on the icon: the hover opens it again");
}

/// The flyout takes the pointer over itself — egui's, so the 3D view does not orbit through it —
/// and only there: the 3D view beside it keeps its input; the strip is egui's; the view the frame
/// hands the viewer is what the strip leaves.
#[test]
fn the_flyout_takes_the_pointer_and_the_3d_view_keeps_the_rest() {
    use super::workspace::layout;
    use DockTab::Graph;
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    st.workspace.set_node(layout::parse("H(0.30,[Model*],H(0.70,3D,[Graph*]))").unwrap());
    for _ in 0..2 {
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
    }
    st.workspace.pin(Graph, false);
    st.open_tab(Graph);
    let mut view = egui::Rect::NOTHING;
    for _ in 0..12 {
        view = frame(&ctx, &mut st, &inp, 1400.0, vec![]).view.unwrap();
    }
    let (fly, strip) = (probed_rect("flyout-Graph"), probed_rect("strip-Right"));
    assert!((fly.right() - view.right()).abs() < 1.0 && (strip.left() - view.right()).abs() < 1.0, "the flyout against the right strip: {fly:?} {strip:?} {view:?}");
    let over = |st: &mut UiState, at: egui::Pos2| {
        frame(&ctx, st, &inp, 1400.0, vec![egui::Event::PointerMoved(at)]);
        (ctx.is_pointer_over_egui(), ctx.egui_wants_pointer_input())
    };
    assert_eq!(over(&mut st, fly.center()), (true, true), "the flyout is egui's");
    assert_eq!(over(&mut st, egui::pos2(fly.left() + 4.0, fly.bottom() - 4.0)), (true, true), "all of it");
    assert_eq!(over(&mut st, egui::pos2(fly.left() - 60.0, fly.center().y)), (false, false), "the 3D view beside it is not");
    assert_eq!(over(&mut st, strip.center()), (true, true), "the strip is egui's");
    assert_eq!(st.workspace.flyout(), Some(Graph), "opened by open_tab: still out");
}

/// A leaf's tab bar ends with its pin, left of the close button: clicking where the Pin icon is
/// painted unpins the leaf's active tab (only it); the flyout's close button closes the panel.
#[test]
fn the_tab_bar_pin_unpins_the_active_tab() {
    use super::workspace::layout::{self, Edge};
    use DockTab::*;
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    st.workspace.set_node(layout::parse("H(0.70,3D,V(0.50,[Graph*,Orbitals],[Log*]))").unwrap());
    for _ in 0..2 {
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
    }
    let (pin, pop, close, graph) = (probed_rect("dock-pin-Graph"), probed_rect("dock-pop-Graph"), probed_rect("dock-close-Graph"), probed_rect("dock-tab-Graph"));
    // the bar ends pin, pop out, close (spec D-4)
    assert!(pin.left() > graph.right() && (pin.right() - pop.left()).abs() < 0.5 && (pop.right() - close.left()).abs() < 0.5, "pin {pin:?}, pop {pop:?}, close {close:?}");
    assert!((pin.center().y - close.center().y).abs() < 0.5 && (pop.center().y - close.center().y).abs() < 0.5);
    let outs = click_widget(&ctx, &mut st, &inp, "dock-pin-Graph");
    assert!(outs.iter().all(|o| o.actions.is_empty() && o.commands.is_empty()));
    assert_eq!(layout::encode(&st.workspace.to_node()), "H(0.70,3D,V(0.50,[Orbitals*],[Log*]))");
    let h = st.workspace.hidden();
    assert_eq!((h.len(), h[0].tab, h[0].edge), (1, Graph, Edge::Right));
    // its flyout's close button closes it (the Window menu brings it back)
    st.open_tab(Graph);
    for _ in 0..12 {
        frame(&ctx, &mut st, &inp, 1400.0, vec![]); // past the slide
    }
    let outs = click_widget(&ctx, &mut st, &inp, "flyout-close-Graph");
    assert!(outs.iter().all(|o| o.actions.is_empty() && o.commands.is_empty()));
    assert!(st.workspace.hidden().is_empty() && !st.tab_shown(Graph) && !layout::panels(&st.workspace.to_node()).contains(&Graph));
}

/// The profile keeps the auto-hidden panels (`hidden = …`, spec D-8) and reads them back onto
/// their strips; a profile with a layout and no `hidden` line hides nothing; a hidden entry for a
/// panel the layout already places is dropped (its first place wins).
#[test]
fn the_profile_keeps_the_hidden_panels() {
    use super::workspace::layout::{self, Edge};
    use DockTab::*;
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    assert!(profile::capture(&st).contains("\nhidden = \n"), "a fresh profile hides nothing");
    for _ in 0..2 {
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
    }
    st.workspace.pin(Graph, false);
    st.workspace.pin(Readouts, false);
    let text = profile::capture(&st);
    let line = text.lines().find(|l| l.starts_with("hidden = ")).unwrap();
    assert!(line.starts_with("hidden = Graph:right:") && line.contains(", Readouts:left:"), "{line}");
    let mut back = UiState::new("h2", &scene);
    profile::apply(&mut back, &text);
    assert_eq!(back.workspace.to_node(), st.workspace.to_node());
    assert_eq!(layout::encode_hidden(back.workspace.hidden()), layout::encode_hidden(st.workspace.hidden()));
    back.open_tab(Graph);
    assert!(back.tab_shown(Graph), "reachable: open_tab opens its flyout");
    for _ in 0..2 {
        frame(&ctx, &mut back, &inp, 1400.0, vec![]);
    }
    probed_rect("strip-icon-Graph");
    probed_rect("strip-icon-Readouts");
    // a layout line without a hidden line: nothing hidden
    let mut s = UiState::new("h2", &scene);
    s.workspace.set_hidden(vec![layout::Hidden { tab: Graph, edge: Edge::Left, size: 300.0 }]);
    profile::apply(&mut s, &format!("layout = {}\n", layout::encode(&st.workspace.to_node())));
    assert!(s.workspace.hidden().is_empty());
    // a hidden panel the layout places: the layout's
    profile::apply(&mut s, &format!("layout = {}\nhidden = Log:left:300, Graph:bottom:250\n", layout::encode(&layout::default_layout())));
    assert_eq!(s.workspace.hidden(), &[]);
    profile::apply(&mut s, "layout = H(0.70,3D,[Log*])\nhidden = log:left:300, GRAPH:bottom:5000, Bogus:left:300\n");
    assert_eq!(s.workspace.hidden(), &[layout::Hidden { tab: Graph, edge: Edge::Bottom, size: layout::HIDDEN_MAX }]);
}

/// Deferred finding (docking D1 review): `egui_dock` clamps each split's fraction to leave 80 pt
/// either side, which below ≈ 333 pt rewrote the default's 24 % (and the autosave kept it). At
/// 300 pt wide the default's encoding is unchanged after a few frames — and a separator drag at
/// full width still moves its split.
#[test]
fn a_very_narrow_window_keeps_the_saved_fractions() {
    use super::workspace::layout;
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    let before = layout::encode(&st.workspace.to_node());
    for _ in 0..4 {
        frame(&ctx, &mut st, &inp, 300.0, vec![]);
    }
    assert_eq!(layout::encode(&st.workspace.to_node()), before, "at 300 pt wide");
    for _ in 0..2 {
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
    }
    assert_eq!(layout::encode(&st.workspace.to_node()), before, "and back at 1400");
    // the left column's separator, 24 % of 1400, dragged 100 pt right
    let at = egui::pos2(336.0, 600.0);
    frame(&ctx, &mut st, &inp, 1400.0, vec![egui::Event::PointerMoved(at), pointer(at, true)]);
    for i in 1..=4 {
        frame(&ctx, &mut st, &inp, 1400.0, vec![egui::Event::PointerMoved(at + egui::vec2(25.0 * i as f32, 0.0))]);
    }
    frame(&ctx, &mut st, &inp, 1400.0, vec![pointer(at + egui::vec2(100.0, 0.0), false)]);
    let layout::Node::Split { frac, .. } = st.workspace.to_node() else { panic!() };
    assert!((0.29..0.33).contains(&frac), "dragged to ≈ 436 / 1400: {frac}");
    // at 300 pt the saved 24 % lies below egui_dock's bound (80 / 300): a slow drag (2 pt a frame)
    // of that separator still moves it — the clamp is not undone while it is dragged
    st.reset_layout();
    for _ in 0..3 {
        frame(&ctx, &mut st, &inp, 300.0, vec![]);
    }
    let at = egui::pos2(0.24 * 300.0, 600.0);
    frame(&ctx, &mut st, &inp, 300.0, vec![egui::Event::PointerMoved(at), pointer(at, true)]);
    for i in 1..=12 {
        frame(&ctx, &mut st, &inp, 300.0, vec![egui::Event::PointerMoved(at + egui::vec2(2.0 * i as f32, 0.0))]);
    }
    frame(&ctx, &mut st, &inp, 300.0, vec![pointer(at + egui::vec2(24.0, 0.0), false)]);
    frame(&ctx, &mut st, &inp, 300.0, vec![egui::Event::PointerGone]);
    let layout::Node::Split { frac, .. } = st.workspace.to_node() else { panic!() };
    assert!(frac > 0.27, "a slow drag at 300 pt moves the separator right: {frac}");
}

/// Every panel draws in its flyout (no panic, no command), on each edge.
#[test]
fn every_panel_draws_in_its_flyout() {
    use super::workspace::layout::{self, Edge, Hidden};
    let scene = Scene::preset("h2o").unwrap();
    let cam = camera();
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2o", &scene);
    for snap in [None, Some(snapshot(&scene, Backend::Gpu, None)), Some(snapshot(&scene, Backend::Estimated, Some(model())))] {
        let inp = input(&scene, snap.as_ref(), &cam);
        for (i, t) in DockTab::PANELS.into_iter().enumerate() {
            st.workspace.set_node(layout::parse("H(0.70,3D,[Log*])").unwrap());
            if t == DockTab::Log {
                st.workspace.close_panel(t);
            }
            st.workspace.set_hidden(vec![Hidden { tab: t, edge: Edge::ALL[i % 3], size: 360.0 }]);
            st.open_tab(t);
            for _ in 0..3 {
                let out = frame(&ctx, &mut st, &inp, 1400.0, vec![]);
                assert!(out.commands.is_empty() && out.actions.is_empty(), "{t:?}");
            }
            assert!(st.tab_shown(t), "{t:?}");
            probed_rect(&format!("flyout-{t:?}"));
        }
    }
}

// ---- the Play snapshot (`ui::rewind`) ----

/// Play from paused stores the scene (velocities and isotopes included), the driver's pins, the
/// run settings and the time; Play while playing, Play on the estimated solver without a model and
/// an equal state played again store nothing new; the newest `rewind::KEPT` are kept.
#[test]
fn pressing_play_keeps_a_snapshot() {
    let mut scene = Scene::preset("h2").unwrap();
    scene.nuclei[0].vel = [1e-4, 0.0, 0.0];
    scene.nuclei[1].a = Some(2);
    let cam = camera();
    let mut snap = snapshot(&scene, Backend::Gpu, None);
    snap.dynamics.pins = vec![false, true];
    snap.dynamics.sim_time = units::fs_to_au_time(12.5);
    snap.dynamics.step = 50;
    let inp = input(&scene, Some(&snap), &cam);
    let mut st = UiState::new("h2", &scene);
    st.gravity = true;
    let out = act(&mut st, &inp, Action::TogglePlay);
    assert!(out.toggle_play && out.commands.is_empty());
    assert_eq!(st.play_snapshots.len(), 1);
    let s = &st.play_snapshots[0];
    assert_eq!(s.state.scene, scene, "positions, velocities and isotopes");
    assert_eq!(s.state.pins, [false, true]);
    assert!(s.state.gravity && s.state.walls_on);
    assert_eq!((s.step, (s.sim_time_fs - 12.5).abs() < 1e-9), (50, true));
    assert!(matches!(st.activity.timeline.last(), Some(activity::Mark::Snapshot(_, t)) if (t - 12.5).abs() < 1e-9));
    // the same state played again: no copy
    act(&mut st, &inp, Action::Play);
    assert_eq!(st.play_snapshots.len(), 1);
    // while playing: nothing
    let mut playing = input(&scene, Some(&snap), &cam);
    playing.playing = true;
    let mut moved = scene.clone();
    moved.nuclei[0].pos[0] += 0.1;
    playing.scene = &moved;
    act(&mut st, &playing, Action::TogglePlay);
    assert_eq!(st.play_snapshots.len(), 1);
    // the estimated solver without a model refuses Play: nothing
    let est = snapshot(&moved, Backend::Estimated, None);
    let inp_est = input(&moved, Some(&est), &cam);
    let mut st_est = UiState::new("h2", &scene);
    st_est.backend = Backend::Estimated;
    act(&mut st_est, &inp_est, Action::Play);
    assert!(st_est.play_snapshots.is_empty());
    // newest first, at most KEPT
    let scenes: Vec<Scene> = (0..rewind::KEPT + 2)
        .map(|k| {
            let mut s = scene.clone();
            s.nuclei[0].pos[0] += 0.01 * (k + 1) as f64;
            s
        })
        .collect();
    for s in &scenes {
        act(&mut st, &input(s, None, &cam), Action::TogglePlay);
    }
    assert_eq!(st.play_snapshots.len(), rewind::KEPT);
    assert_eq!(st.play_snapshots[0].state.scene, *scenes.last().unwrap());
    // playing an older one again makes it the newest (Rewind returns to it), still no copy
    let older = st.play_snapshots[2].state.scene.clone();
    act(&mut st, &input(&older, None, &cam), Action::TogglePlay);
    assert_eq!(st.play_snapshots.len(), rewind::KEPT);
    assert_eq!(st.play_snapshots[0].state.scene, older);
}

/// Rewind puts the snapshot back through the scene-replacement path: Pause, SetScene with the
/// snapshot's velocities first, then the field, COM correction, walls, a changed thermostat and
/// the pins; the panel's settings follow; the timeline marks it. Without a snapshot it is an
/// error and sends nothing.
#[test]
fn rewind_restores_the_play_snapshot() {
    let cam = camera();
    let mut st = UiState::new("h2", &Scene::preset("h2").unwrap());
    let scene = Scene::preset("h2").unwrap();
    let out = act(&mut st, &input(&scene, None, &cam), Action::RewindTo(0));
    assert!(out.commands.is_empty() && out.error.as_deref() == Some(rewind::NONE_YET), "{:?}", out.error);

    let mut at_play = scene.clone();
    at_play.nuclei[0].vel = [0.0, 2e-4, 0.0];
    let mut snap = snapshot(&at_play, Backend::Gpu, None);
    snap.dynamics.pins = vec![true, false];
    st.walls_on = true;
    act(&mut st, &input(&at_play, Some(&snap), &cam), Action::TogglePlay);
    // the run goes on: the scene moves, settings change
    let mut later = at_play.clone();
    later.nuclei[0].pos = [3.0, 3.0, 3.0];
    st.walls_on = false;
    st.gravity = true;
    st.thermostat_on = true;
    let out = act(&mut st, &input(&later, None, &cam), Action::RewindTo(0));
    assert!(out.error.is_none(), "{:?}", out.error);
    let c = &out.commands;
    assert!(matches!(&c[0], Command::Dyn(DynCommand::Pause)), "{c:?}");
    assert!(matches!(&c[1], Command::SetScene(s) if *s == at_play), "then the scene, velocities kept: {c:?}");
    assert!(c.iter().any(|c| matches!(c, Command::Dyn(DynCommand::SetExternalField(a)) if *a == [0.0; 3])), "{c:?}");
    assert!(c.iter().any(|c| matches!(c, Command::Dyn(DynCommand::SetWalls(Some(_))))), "{c:?}");
    assert!(c.iter().any(|c| matches!(c, Command::Dyn(DynCommand::SetComCorrection(true)))), "{c:?}");
    assert!(c.iter().any(|c| matches!(c, Command::Dyn(DynCommand::SetThermostat(None)))), "the thermostat was off at Play: {c:?}");
    let pins: Vec<_> = c.iter().filter_map(|c| if let Command::Dyn(DynCommand::Pin(i, on)) = c { Some((*i, *on)) } else { None }).collect();
    assert_eq!(pins, [(0, true)]);
    assert!(st.walls_on && !st.gravity && !st.thermostat_on, "the panel follows the snapshot");
    assert!(matches!(st.activity.timeline.last(), Some(activity::Mark::Restore(..))));
    assert!(st.status.contains("rewound"), "{}", st.status);
    // an unchanged thermostat is not re-sent (it would redraw velocities at rest)
    let out = act(&mut st, &input(&later, None, &cam), Action::RewindTo(0));
    assert!(!out.commands.iter().any(|c| matches!(c, Command::Dyn(DynCommand::SetThermostat(_)))), "{:?}", out.commands);
    // a snapshot that is not there
    let out = act(&mut st, &input(&later, None, &cam), Action::RewindTo(3));
    assert!(out.commands.is_empty() && out.error.as_deref().is_some_and(|e| e.contains("1 kept")), "{:?}", out.error);
}

/// The toolbar's Rewind is disabled until Play was pressed; the old v → −v is Physics ▸ Reverse
/// velocities and both are found by Ctrl K.
#[test]
fn rewind_and_reverse_velocities_have_their_homes() {
    let names: Vec<&str> = search::ENTRIES.iter().map(|e| e.name).collect();
    assert!(names.iter().any(|n| n.starts_with("Rewind")));
    let rev = search::ENTRIES.iter().find(|e| e.name == "Reverse velocities").expect("searchable");
    assert_eq!(rev.place, search::Place::Menu(menus::MenuId::Physics));
    assert!(rewind::ago(std::time::Duration::from_secs(130)) == "2 min ago");
}

/// The Model tab's amber lone-atom note (`ModelStatus::lone_note`) is drawn by Start sampling and
/// sends nothing; without a note none is drawn.
#[test]
fn the_model_tab_warns_of_atoms_without_a_neighbour() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let note = "1 of 3 atoms has no neighbour within 6 Bohr — it won't be learned; move it within 6 Bohr (≈ 3.2 Å) of another atom";
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    st.backend = Backend::Estimated;
    st.open_tab(DockTab::Model);
    let without = snapshot(&scene, Backend::Estimated, Some(model()));
    let inp = input(&scene, Some(&without), &cam);
    PROBES.with(|p| p.borrow_mut().clear());
    for _ in 0..3 {
        assert!(frame(&ctx, &mut st, &inp, 1400.0, vec![]).commands.is_empty());
    }
    assert!(PROBES.with(|p| p.borrow().get("lone-note").is_none()), "no note without lone atoms");
    let with = snapshot(&scene, Backend::Estimated, Some(ModelStatus { lone_note: Some(note.into()), ..model() }));
    let inp = input(&scene, Some(&with), &cam);
    for _ in 0..3 {
        assert!(frame(&ctx, &mut st, &inp, 1400.0, vec![]).commands.is_empty());
    }
    assert!(probed_rect("lone-note").width() > 0.0, "the note is drawn by Start sampling");
}

/// M8a user-fix 3, the controller's ruling on concern 1: a class refused for one short key names
/// that key and its floor, in the Activity tab and the Model tab alike, even when the class holds
/// only that key (rows per key used to appear only for classes of two or more keys).
#[test]
fn a_refused_class_names_its_short_key() {
    let nn = sim::SpeciesVerdict { z: 7, neighbours: vec![7], trusted: false, refusal: Some((sim::ClassKey::Samples, 5.0, 29.0)), calibration: 1.0, measured: None, measured_energy: None, worst_atom: None, evidence: 0, trusted_since: None, inflation: 1.0, samples: 5, columns: 29 };
    let pair = sim::ClassLoo { class: "pair", samples: 5, force_rms: 0.1, force_max: 0.2, energy_rms: 0.1, energy_max: 0.2, trusted: false, refusal: nn.refusal, calibration: 1.0, measured: None, measured_energy: None, worst_atom: None, evidence: 0, trusted_since: None, inflation: 1.0, species: vec![nn.clone()] };
    let m = ModelStatus { class_loo: vec![pair.clone()], ..model() };
    let scene = Scene::preset("h2").unwrap();
    let st = UiState::new("h2", &scene);
    let lines = activity::trust::lines(&m, &scene, &st);
    assert!(
        lines.iter().any(|l| l.text.contains("around N, with N") && l.text.contains("5 of 29 samples needed")),
        "{:?}",
        lines.iter().map(|l| &l.text).collect::<Vec<_>>()
    );
    assert_eq!(dock::model::species_line("pair", &nn), "  pair around N, with N: not learned yet (DFT) · 5 of 29 samples needed");
    // a trusted single-key class keeps its one line
    let ok = sim::SpeciesVerdict { trusted: true, refusal: None, samples: 40, ..nn };
    let m = ModelStatus { class_loo: vec![sim::ClassLoo { trusted: true, refusal: None, samples: 40, species: vec![ok], ..pair }], ..model() };
    assert!(!activity::trust::lines(&m, &scene, &st).iter().any(|l| l.text.contains("around N, with N")));
}

/// M8a Task 9: the knowledge map — the tab draws a map, a click picks a pair or an element, the pair
/// detail names the environment keys behind its verdict, and the overlay is display state kept in
/// the profile.
mod knowledge_map {
    use super::*;
    use knowledge::Pick;
    use std::sync::Arc;

    fn texts(ctx: &egui::Context, st: &mut UiState, inp: &PanelInput<'_>) -> String {
        fn walk(s: &egui::epaint::Shape, out: &mut String) {
            match s {
                egui::epaint::Shape::Text(t) => {
                    out.push_str(t.galley.text());
                    out.push('\n');
                }
                egui::epaint::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                _ => {}
            }
        }
        let mut out = String::new();
        for _ in 0..3 {
            let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0))), ..Default::default() };
            let mut full = ctx.run_ui(raw, |root| {
                let o = show(root, st, inp);
                assert!(o.commands.is_empty() && o.actions.is_empty(), "the map is display only");
            });
            full.textures_delta.clear();
            out.clear();
            full.shapes.iter().for_each(|c| walk(&c.shape, &mut out));
        }
        out
    }

    /// Three carbons in a chain with two hydrogens (C–C 2.89 Bohr, C–H 2.06 Bohr): a C–C the
    /// fixture map has never seen.
    fn chain() -> Scene {
        let n = |z: u8, pos: [f64; 3]| crate::scene::SceneNucleus { z, a: None, pos, vel: [0.0; 3] };
        Scene {
            nuclei: vec![n(6, [-2.89, 0.0, 0.0]), n(6, [0.0, 0.0, 0.0]), n(6, [2.89, 0.0, 0.0]), n(1, [0.0, 2.06, 0.0]), n(1, [-2.89, -2.06, 0.0])],
            ..Scene::preset("h2o").unwrap()
        }
    }

    fn with_map(scene: &Scene, m: ModelStatus) -> Snapshot {
        let mut snap = snapshot(scene, Backend::Estimated, Some(m));
        snap.activity.knowledge = Some(Arc::new(knowledge::fixture_map()));
        snap
    }

    #[test]
    fn the_knowledge_tab_draws_the_map_and_picks_by_click() {
        let scene = chain();
        let cam = camera();
        let snap = with_map(&scene, model());
        let inp = input(&scene, Some(&snap), &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2o", &scene);
        st.backend = Backend::Estimated;
        st.activity.absorb(&snap);
        assert!(st.activity.knowledge.is_some(), "the map is kept from the snapshot");
        alone(&mut st, DockTab::Knowledge);
        let t = texts(&ctx, &mut st, &inp);
        for want in ["lsda-h0.15-s0.15-tol1e-6-gpu-accepted-3f9a", "2140 samples · H C O", "3 trusted", "2 thin", "1 never seen", "C–C · never", "C–H · trusted", "H–H · thin", "2140 atoms", "THIS SCENE ASKS"] {
            assert!(t.contains(want), "{want:?} in the tab:\n{t}");
        }
        // nothing picked: the weakest pair the scene asks (C–C, never seen) and its empty histogram
        assert!(t.contains("the weakest pair this scene asks") && t.contains("nothing taught") && t.contains("never taught"), "{t}");
        // a click on a line's label picks the pair: its histogram with the scene's distances
        let outs = click_widget(&ctx, &mut st, &inp, "knowledge-pair-1-6");
        assert!(outs.iter().all(|o| o.actions.is_empty() && o.commands.is_empty()), "display only");
        assert_eq!(st.knowledge.pick, Pick::Pair(1, 6));
        let t = texts(&ctx, &mut st, &inp);
        assert!(t.contains("the pair you picked · 610 samples") && t.contains("asked now (the open scene):") && t.contains("2 at 2.06 taught"), "{t}");
        // a node picks its element: neighbour counts and bond angles, the scene's atoms marked
        click_widget(&ctx, &mut st, &inp, "knowledge-element-6");
        assert_eq!(st.knowledge.pick, Pick::Element(6));
        let t = texts(&ctx, &mut st, &inp);
        assert!(t.contains("element · 610 atoms in the fitted samples") && t.contains("asked: #0") && t.contains("BOND ANGLES AT C"), "{t}");
        // the pop-out window is gone (docking spec D-5): the panel in a wide leaf (or a detached
        // window) lays the same map out in two columns, keeping the pick
        st.workspace.set_node(super::super::workspace::layout::parse("H(0.25,3D,[Knowledge*])").unwrap());
        let t = texts(&ctx, &mut st, &inp);
        assert!(t.contains("element · 610 atoms in the fitted samples") && t.contains("BOND ANGLES AT C"), "the wide panel keeps the pick:\n{t}");
        // another solver: the empty state
        let gpu = snapshot(&scene, Backend::Gpu, None);
        st.backend = Backend::Gpu;
        let t = texts(&ctx, &mut st, &input(&scene, Some(&gpu), &cam));
        assert!(t.contains("Pick Estimated to see it"), "{t}");
    }

    #[test]
    fn the_pair_detail_lists_the_keys_behind_its_verdict() {
        let scene = chain();
        let cam = camera();
        let key = |z: u8, neighbours: Vec<u8>, refusal: Option<(sim::ClassKey, f64, f64)>, samples: usize, columns: usize| sim::SpeciesVerdict {
            z,
            neighbours,
            trusted: refusal.is_none(),
            refusal,
            calibration: 1.0,
            measured: None,
            measured_energy: None,
            worst_atom: None,
            evidence: 0,
            trusted_since: None,
            inflation: 1.0,
            samples,
            columns,
        };
        let mut m = model();
        let pair = m.class_loo[0].clone();
        m.class_loo = vec![
            sim::ClassLoo { species: vec![key(1, vec![1], None, 120, 12), key(6, vec![1], None, 600, 30)], ..pair.clone() },
            sim::ClassLoo { class: "many-body", trusted: false, species: vec![key(1, vec![1, 7], Some((sim::ClassKey::Samples, 3.0, 92.0)), 3, 92), key(7, vec![1], None, 120, 20)], ..pair },
        ];
        let snap = with_map(&scene, m.clone());
        let inp = input(&scene, Some(&snap), &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2o", &scene);
        st.backend = Backend::Estimated;
        st.activity.absorb(&snap);
        alone(&mut st, DockTab::Knowledge);
        st.knowledge.pick = Pick::Pair(1, 1);
        let t = texts(&ctx, &mut st, &inp);
        assert!(t.contains("thin: taught, but not everywhere it was met"), "{t}");
        assert!(t.contains("Three-or-more-atom arrangements around H, with H and N — not yet (3 of 92 samples needed)"), "the short key, in the Activity tab's words:\n{t}");
        // only the keys where H sees H: not C's or N's (the Activity tab lists those elsewhere)
        assert!(t.contains("learned: pair around H, with H\n"), "{t}");
        assert!(!t.contains("the model as a whole is not trusted"), "{t}");
        // an untrusted model: rule 4 is said first
        let untrusted = ModelStatus { trusted: false, untrusted_reason: Some("max training energy residual 2.2e-3 > 1.0e-3 Ha/atom".into()), ..m };
        let snap = with_map(&scene, untrusted);
        let t = texts(&ctx, &mut st, &input(&scene, Some(&snap), &cam));
        assert!(t.contains("the model as a whole is not trusted yet (max training energy residual"), "{t}");
    }

    #[test]
    fn the_knowledge_overlay_is_display_only_and_kept_in_the_profile() {
        let scene = chain();
        let cam = camera();
        let snap = with_map(&scene, model());
        let inp = input(&scene, Some(&snap), &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2o", &scene);
        st.backend = Backend::Estimated;
        st.activity.absorb(&snap);
        assert!(!st.show_knowledge, "off by default");
        assert!(knowledge::overlay_coverage(&st, &scene.nuclei).is_empty(), "off: no tint");
        // View ▸ Display aids ▸ knowledge overlay: no command, no recorded action
        super::view_panel(&mut st, &[crate::ui::dock::view::Section::Aids]);
        let outs = click_widget(&ctx, &mut st, &inp, "show-knowledge");
        assert!(outs.iter().all(|o| o.commands.is_empty() && o.actions.is_empty()), "display only");
        assert!(st.show_knowledge);
        // the tint: the carbons see C–C, never seen
        let cov = knowledge::overlay_coverage(&st, &scene.nuclei);
        assert_eq!(cov.len(), 5);
        assert_eq!(&cov[..3], &[0.0, 0.0, 0.0], "{cov:?}");
        assert_eq!(knowledge::tint_srgb(0.0), [122, 118, 144]);
        assert_eq!(knowledge::tint_srgb(0.5), [228, 192, 68]);
        assert_eq!(knowledge::tint_srgb(1.0), [110, 210, 120]);
        let t = texts(&ctx, &mut st, &inp);
        assert!(t.contains("knowledge overlay") && t.contains("1 well known") && t.contains("per atom: its least-known neighbour within 6.0 Bohr"), "the legend:\n{t}");
        // Ctrl K finds it in the View panel, and the map in its own
        assert_eq!(search::results("knowledge overlay", 1)[0].place, search::Place::Dock(DockTab::View));
        assert_eq!(search::results("knowledge map", 1)[0].place, search::Place::Dock(DockTab::Knowledge));
        // the profile keeps it
        let text = profile::capture(&st);
        assert!(text.contains("show_knowledge = true"), "{text}");
        let mut fresh = UiState::new("h2o", &scene);
        profile::apply(&mut fresh, &text);
        assert!(fresh.show_knowledge);
        // a profile saved before the workspace (and before the Knowledge tab) migrates into it
        // (docking spec D-8): its docks' tabs in the matching columns, every panel once, the tab
        // each dock showed active
        use super::super::workspace::layout;
        let once = |s: &UiState| {
            let p = layout::panels(&s.workspace.to_node());
            DockTab::PANELS.into_iter().all(|t| p.iter().filter(|&&x| x == t).count() == 1) && p.len() == DockTab::PANELS.len()
        };
        let mut old = UiState::new("h2o", &scene);
        profile::apply(&mut old, "left_tabs = Activity Readouts Graph Log\nright_tabs = Model Orbitals Selection\nright_tab = Selection\nleft_tab = Graph\nleft_width = 400\ndock_open = false\n");
        assert!(once(&old), "{}", layout::encode(&old.workspace.to_node()));
        assert_eq!(
            layout::encode(&old.workspace.to_node()),
            "H(0.24,[Activity,Readouts,Graph*,Log],H(0.70,3D,[Model,View,Orbitals,Selection*,Knowledge]))",
            "the moved Log stays left; Knowledge, never listed, joins the right column"
        );
        assert!(old.tab_shown(DockTab::Selection) && old.tab_shown(DockTab::Graph));
        // a tab named twice keeps its first place (it was ignored by the old docks' reader)
        let mut dup = UiState::new("h2o", &scene);
        profile::apply(&mut dup, "left_tabs = Activity Readouts Graph Log\nright_tabs = Model Log\n");
        assert!(once(&dup), "{}", layout::encode(&dup.workspace.to_node()));
        assert!(layout::encode(&dup.workspace.to_node()).starts_with("H(0.24,[Activity*,Readouts,Graph,Log],"), "{}", layout::encode(&dup.workspace.to_node()));
        // only one of the two lists: the other column gets the panels it does not name
        let mut half = UiState::new("h2o", &scene);
        profile::apply(&mut half, "right_tabs = Log Model\n");
        assert!(once(&half), "{}", layout::encode(&half.workspace.to_node()));
    }
}

/// Docking Task 4 (spec D-6, D-7): the View panel holds the View menu's controls in five folding
/// sections, and the Window menu lists the panels.
mod view_panel {
    use super::super::dock::view::{Section, Sections};
    use super::*;

    /// The View menu's Ctrl K entries before the panel (27): each lives in the View panel now.
    const FORMER: [&str; 27] = [
        "Density", "Orbitals", "All orbitals coloured", "Spin density", "Colour dots by density / orbital", "Density: volume / isosurface", "Gamma",
        "Opacity", "Iso level (× ρ max)", "Refresh cloud every", "Atomic superposition between DFT clouds", "Cell box", "Floor grid", "Axes",
        "Placement plane", "Grid spacing 1 Bohr / 1 Å", "Partial core density", "Smooth motion", "Show what training is solving",
        "Knowledge overlay", "Trails", "Bonds", "Molecule names", "Name style", "Composition badge on hover", "Bond angles", "Partial charges",
    ];

    /// Ctrl K: every View-panel entry — the 27 former View-menu ones and the panel's own — shows the
    /// View panel (from behind another tab, every section folded), unfolds the section holding its
    /// anchor and scrolls the anchor's group into the panel's visible part.
    #[test]
    fn every_former_view_menu_entry_reveals_inside_the_view_panel() {
        for name in FORMER {
            assert!(search::ENTRIES.iter().any(|e| e.name == name && e.place == search::Place::Dock(DockTab::View)), "{name}");
        }
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let snap = snapshot(&scene, Backend::Estimated, Some(model()));
        let inp = input(&scene, Some(&snap), &cam);
        let entries: Vec<&search::Entry> = search::ENTRIES.iter().filter(|e| e.place == search::Place::Dock(DockTab::View)).collect();
        assert!(entries.len() > FORMER.len());
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0));
        for e in entries {
            let ctx = egui::Context::default();
            apply_theme(&ctx);
            let mut st = UiState::new("h2", &scene);
            st.backend = Backend::Estimated;
            st.open_tab(DockTab::Orbitals); // the View panel behind it
            st.view_sections = Sections::all(false);
            frame(&ctx, &mut st, &inp, 1400.0, vec![]);
            assert!(!st.tab_shown(DockTab::View));
            search::pick(&ctx, &mut st, e);
            for i in 0..20 {
                if i == 19 {
                    PROBES.with(|p| p.borrow_mut().clear()); // only the last frame's rects
                }
                let raw = egui::RawInput { screen_rect: Some(screen), time: Some(100.0 + 0.05 * i as f64), ..Default::default() };
                let mut full = ctx.run_ui(raw, |root| {
                    show(root, &mut st, &inp);
                });
                full.textures_delta.clear();
            }
            assert!(st.tab_shown(DockTab::View), "{}: the View panel is shown", e.name);
            if let Some(s) = Section::ALL.into_iter().find(|s| s.anchor() == e.anchor) {
                assert!(st.view_sections.is_open(s), "{}: its section {s:?} unfolds", e.name);
                assert_eq!(Section::ALL.into_iter().filter(|&x| st.view_sections.is_open(x)).count(), 1, "{}: only its own", e.name);
            }
            let (a, panel) = (probed_rect(&format!("anchor-{}", e.anchor)), probed_rect("view-panel"));
            assert!(panel.intersects(a) && a.left() >= panel.left() - 1.0 && a.right() <= panel.right() + 1.0, "{}: anchor {a:?} inside the View panel {panel:?}", e.name);
        }
    }

    /// Hidden, not greyed: the volume's sliders only as a volume, the iso level only as an
    /// isosurface, the dots' look only as dots (their orbital colouring only under All orbitals
    /// coloured), the Camera view's settings only while it is on, the cloud's estimated-only
    /// options only under that solver, bloom's threshold only with bloom.
    #[test]
    fn controls_that_do_not_apply_are_hidden() {
        fn texts(ctx: &egui::Context, st: &mut UiState, inp: &PanelInput<'_>) -> String {
            fn walk(s: &egui::epaint::Shape, out: &mut String) {
                match s {
                    egui::epaint::Shape::Text(t) => {
                        out.push_str(t.galley.text());
                        out.push('\n');
                    }
                    egui::epaint::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                    _ => {}
                }
            }
            let mut out = String::new();
            for _ in 0..2 {
                let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 1600.0))), ..Default::default() };
                let mut full = ctx.run_ui(raw, |root| {
                    show(root, st, inp);
                });
                full.textures_delta.clear();
                out.clear();
                full.shapes.iter().for_each(|c| walk(&c.shape, &mut out));
            }
            out
        }
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        view_panel(&mut st, &Section::ALL);
        st.volume_mode = VOLUME_VOLUME;
        st.bloom = 0.0;
        let t = texts(&ctx, &mut st, &inp);
        assert!(t.contains("gamma") && !t.contains("iso (× ρ max)") && !t.contains("dot size"), "{t}");
        assert!(!t.contains("refresh cloud every") && t.contains("cloud options apply to the estimated solver"), "{t}");
        assert!(!t.contains("exposure") && !t.contains("bloom from"), "{t}");
        st.volume_mode = VOLUME_ISO;
        st.camera_view = true;
        st.backend = Backend::Estimated;
        st.bloom = 0.5;
        let t = texts(&ctx, &mut st, &inp);
        assert!(!t.contains("gamma") && t.contains("iso (× ρ max)") && t.contains("exposure") && t.contains("refresh cloud every") && t.contains("bloom from"), "{t}");
        st.volume_mode = VOLUME_DOTS;
        let t = texts(&ctx, &mut st, &inp);
        assert!(t.contains("dot size") && !t.contains("colour dots by"), "{t}");
        st.show = ShowMode::Coloured;
        assert!(texts(&ctx, &mut st, &inp).contains("colour dots by"));
        // a folded section draws nothing of its own
        st.view_sections = Sections::all(false);
        let t = texts(&ctx, &mut st, &inp);
        assert!(t.contains("What you see") && t.contains("Display aids") && !t.contains("dot size") && !t.contains("partial charges"), "{t}");
    }

    /// Drawing the panel never rewrites a stored value outside a slider's range (the review of
    /// Task 4: egui clamps on every draw by default): a profile's (or `--ui nuclei=`'s) nucleus
    /// size of 3.0 stays 3.0. (The dots' feather was already held to 0–0.8 by the view switch's
    /// own slider, drawn whenever the dots are.)
    #[test]
    fn drawing_does_not_clamp_stored_values() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        view_panel(&mut st, &[]);
        profile::apply(&mut st, "nucleus_scale = 3\nview_sections = aids\n");
        for _ in 0..3 {
            frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        }
        probed_rect("nucleus-auto"); // the Display aids were drawn
        assert_eq!(st.nucleus_scale, Some(3.0));
        assert!(profile::capture(&st).contains("\nnucleus_scale = 3\n"));
    }

    /// The sections fold and unfold by their headers; their open state is kept in the profile
    /// (`view_sections`), What you see and How it looks open in a profile without it.
    #[test]
    fn sections_fold_and_the_profile_keeps_them() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        assert_eq!(st.view_sections, Sections::default());
        assert!(profile::capture(&st).contains("\nview_sections = what,looks\n"));
        alone(&mut st, DockTab::View);
        let outs = click_widget(&ctx, &mut st, &inp, "view-bonds");
        assert!(outs.iter().all(|o| o.actions.is_empty() && o.commands.is_empty()), "display state");
        click_widget(&ctx, &mut st, &inp, "view-looks");
        assert_eq!(st.view_sections.encode(), "what,bonds");
        let text = profile::capture(&st);
        let mut fresh = UiState::new("h2", &scene);
        profile::apply(&mut fresh, &text);
        assert_eq!(fresh.view_sections.encode(), "what,bonds");
        profile::apply(&mut fresh, "gamma = 0.5\n");
        assert_eq!(fresh.view_sections, Sections::default(), "absent: the default two");
        profile::apply(&mut fresh, "view_sections = \n");
        assert_eq!(fresh.view_sections, Sections::all(false), "empty: every section folded");
    }

    /// Reset view settings: the look, Camera, Bonds & labels and Display aids back to their
    /// defaults — the settings that are actions by their actions (recorded), the rest directly —
    /// and What you see left alone.
    #[test]
    fn reset_view_settings_puts_back_the_defaults() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        let d = UiState::new("h2", &scene);
        view_panel(&mut st, &[]);
        (st.gamma, st.camera_view, st.show_cell, st.label_opacity, st.nucleus_scale, st.background) = (0.3, true, !d.show_cell, 0.2, Some(1.5), Some([0, 0, 0]));
        st.show_bonds = !d.show_bonds;
        st.hide_cloud = true;
        let outs = click_widget(&ctx, &mut st, &inp, "view-reset");
        let acts: Vec<&Action> = outs.iter().flat_map(|o| &o.actions).collect();
        assert!(matches!(acts.as_slice(), [Action::ShowBonds(b)] if *b == d.show_bonds), "{acts:?}");
        assert_eq!((st.gamma, st.camera_view, st.show_cell, st.label_opacity, st.nucleus_scale, st.background), (d.gamma, d.camera_view, d.show_cell, d.label_opacity, None, None));
        assert_eq!(st.show_bonds, d.show_bonds);
        assert!(st.hide_cloud, "What you see stays");
        assert_eq!(st.view_sections, Sections::all(false), "the folds stay");
    }

    /// The Window menu (spec D-7): one row per panel, a click shows it (a closed one comes back);
    /// Reset layout gives the default; none of it is an action. The View menu is gone.
    #[test]
    fn the_window_menu_shows_panels_and_resets_the_layout() {
        use super::super::workspace::layout;
        assert_eq!(menus::MenuId::ALL.map(menus::MenuId::title), ["File", "Scene", "Physics", "Window", "Help"]);
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        let in_window = |st: &mut UiState, name: &str| {
            egui::Popup::open_id(&ctx, menus::popup_id(menus::MenuId::Window));
            let outs = click_widget(&ctx, st, &inp, name);
            egui::Popup::close_all(&ctx);
            outs
        };
        let mut outs = Vec::new();
        assert!(!st.tab_shown(DockTab::Graph));
        outs.extend(in_window(&mut st, "window-Graph"));
        assert!(st.tab_shown(DockTab::Graph) && !st.tab_shown(DockTab::Log));
        st.workspace.close_panel(DockTab::Knowledge);
        outs.extend(in_window(&mut st, "window-Knowledge"));
        assert!(st.tab_shown(DockTab::Knowledge), "a closed panel comes back");
        outs.extend(in_window(&mut st, "reset-layout"));
        assert_eq!(layout::encode(&st.workspace.to_node()), layout::encode(&layout::default_layout()));
        assert!(outs.iter().all(|o| o.actions.is_empty() && o.commands.is_empty()), "display state");
        // the screenshot harness: menu=view shows the View panel, menu=window opens the menu
        st.open_tab(DockTab::Orbitals);
        assert_eq!(open_menu(&ctx, &mut st, "view"), Some(()));
        assert!(st.tab_shown(DockTab::View));
        assert_eq!(open_menu(&ctx, &mut st, "Window"), Some(()));
        assert!(egui::Popup::is_id_open(&ctx, menus::popup_id(menus::MenuId::Window)));
        assert_eq!(open_menu(&ctx, &mut st, "bogus"), None);
    }

    /// Window ▸ Layouts ▸ <preset> clicked through the real menus — the top bar's Window button,
    /// the Layouts submenu, the preset's row — in the full layout and in the compact one (where
    /// Window is itself a submenu of "☰ Menu"): each click rearranges the workspace as the preset
    /// (tree and hidden panels), closes every menu, and is no action. The Window menu's check
    /// reads "in the workspace" (shown, behind a tab, or auto-hidden), not "on screen".
    #[test]
    fn window_layouts_applies_a_preset_and_closes_the_menu() {
        use super::super::workspace::{layout, presets::Preset};
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        let arrangement = |st: &UiState| (layout::encode(&st.workspace.to_node()), layout::encode_hidden(st.workspace.hidden()));
        for (w, path) in [(1400.0, &["menu-Window", "layouts-submenu"][..]), (1200.0, &["menu-all", "menu-Window", "layouts-submenu"][..])] {
            st.workspace.set_node(layout::parse("H(0.50,[Log*,Graph],V(0.30,3D,[Model*]))").unwrap());
            for p in [Preset::Training, Preset::Presenting, Preset::TwoMonitors, Preset::Training, Preset::OneMonitor] {
                let before = arrangement(&st);
                let mut outs = Vec::new();
                for step in path.iter().copied().chain([format!("layout-{}", p.word()).as_str()]) {
                    PROBES.with(|m| m.borrow_mut().clear()); // only what this frame drew can be clicked
                    outs.extend(press_widget_w(&ctx, &mut st, &inp, w, step, 0.0));
                }
                // headless: no second monitor known, so Two monitors is One monitor (spec D-11)
                let (tree, hidden) = if p == Preset::TwoMonitors { Preset::OneMonitor.tree() } else { p.tree() };
                assert_eq!(arrangement(&st), (layout::encode(&tree), layout::encode_hidden(&hidden)), "{w}: {p:?}");
                assert_ne!(arrangement(&st), before, "{w}: {p:?} changed the arrangement");
                assert!(!egui::Popup::is_any_open(&ctx), "{w}: {p:?}: the menu closed");
                assert!(outs.iter().all(|o| o.actions.is_empty() && o.commands.is_empty()), "display state");
            }
        }
        // the check: in the workspace, not only on screen
        st.apply_layout(Preset::Presenting);
        // and the profile keeps the preset (its autosave writes what `capture` gives)
        let saved = profile::capture(&st);
        assert!(saved.contains("\nlayout = 3D\n"), "{saved}");
        assert!(saved.contains(&format!("\nhidden = {}\n", layout::encode_hidden(&Preset::Presenting.tree().1))), "{saved}");
        assert!(DockTab::PANELS.into_iter().all(|t| st.workspace.present(t) && !st.tab_shown(t)), "hidden panels are present");
        st.reset_layout();
        assert!(st.workspace.present(DockTab::Graph) && !st.tab_shown(DockTab::Graph), "a tab behind Log is present");
        st.workspace.close_panel(DockTab::Graph);
        assert!(!st.workspace.present(DockTab::Graph), "a closed panel is not");
        // Ctrl K finds each preset (and Reset layout) in the Window menu's Layouts group
        for p in Preset::ALL {
            let r = search::results(&format!("layout {}", p.title()), 1);
            assert_eq!((r[0].name, r[0].anchor, r[0].place), (format!("Layout: {}", p.title()).as_str(), "m-layout", search::Place::Menu(menus::MenuId::Window)));
        }
        assert_eq!(search::results("presenting", 1)[0].name, "Layout: Presenting");
        assert_eq!(search::results("reset layout", 1)[0].name, "Reset layout");
    }
}

/// Detached windows (docking spec D-12, D3): the UI side, headless — the pop-out button, a
/// detached window's own egui context drawn with the same state, docking back, the profile's
/// `window = …` lines, the Two monitors layout with an injected monitor list, and Ctrl K's focus.
mod detached_windows {
    use super::super::workspace::{layout, presets::Preset};
    use super::*;

    const PRIMARY: WinRect = WinRect { x: 0, y: 0, w: 2560, h: 1440 };
    const SECOND: WinRect = WinRect { x: 2560, y: 0, w: 1920, h: 1080 };

    /// One frame of detached window `id` in its own context (`ctx`), 600 × 800 points.
    fn window_frame(ctx: &egui::Context, st: &mut UiState, inp: &PanelInput<'_>, id: u32, events: Vec<egui::Event>) -> PanelOutput {
        let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(600.0, 800.0))), events, ..Default::default() };
        let mut out = None;
        let mut full = ctx.run_ui(raw, |root| out = Some(show_window(root, st, inp, id)));
        full.textures_delta.clear();
        out.unwrap()
    }

    /// Clicks widget `name` in detached window `id` (as `click_widget` does in the main window).
    fn click_in_window(ctx: &egui::Context, st: &mut UiState, inp: &PanelInput<'_>, id: u32, name: &str) -> Vec<PanelOutput> {
        for _ in 0..3 {
            window_frame(ctx, st, inp, id, vec![]);
        }
        let c = probed_rect(name).center();
        vec![
            window_frame(ctx, st, inp, id, vec![egui::Event::PointerMoved(c), pointer(c, true)]),
            window_frame(ctx, st, inp, id, vec![pointer(c, false)]),
            window_frame(ctx, st, inp, id, vec![egui::Event::PointerGone]),
        ]
    }

    fn requests(outs: &[PanelOutput]) -> Vec<WindowRequest> {
        outs.iter().flat_map(|o| o.windows.clone()).collect()
    }

    fn secondary(pos: egui::Pos2, pressed: bool) -> egui::Event {
        egui::Event::PointerButton { pos, button: egui::PointerButton::Secondary, pressed, modifiers: egui::Modifiers::NONE }
    }

    /// Right-clicks tab `t` in the main window (1400 points wide), so its context menu opens, then
    /// clicks the menu's entry `item` (a `tab-menu-*` probe). Every frame's output.
    fn tab_menu(ctx: &egui::Context, st: &mut UiState, inp: &PanelInput<'_>, t: DockTab, item: &str) -> Vec<PanelOutput> {
        for _ in 0..3 {
            frame(ctx, st, inp, 1400.0, vec![]);
        }
        let c = probed_rect(&format!("dock-tab-{t:?}")).center();
        let mut outs = vec![frame(ctx, st, inp, 1400.0, vec![egui::Event::PointerMoved(c), secondary(c, true)]), frame(ctx, st, inp, 1400.0, vec![secondary(c, false)])];
        outs.extend(click_widget(ctx, st, inp, item));
        outs
    }

    /// [`tab_menu`] in detached window `id` (its own context).
    fn tab_menu_in_window(ctx: &egui::Context, st: &mut UiState, inp: &PanelInput<'_>, id: u32, t: DockTab, item: &str) -> Vec<PanelOutput> {
        for _ in 0..3 {
            window_frame(ctx, st, inp, id, vec![]);
        }
        let c = probed_rect(&format!("dock-tab-{t:?}")).center();
        let mut outs = vec![window_frame(ctx, st, inp, id, vec![egui::Event::PointerMoved(c), secondary(c, true)]), window_frame(ctx, st, inp, id, vec![secondary(c, false)])];
        outs.extend(click_in_window(ctx, st, inp, id, item));
        outs
    }

    /// Review M8: the tab menu's move entries, clicked (their `Change`s routed through
    /// `on_tab_button`): Move to new window, Move to window N, then Move to main window from
    /// inside the detached window. None of it is an action.
    #[test]
    fn the_tab_menu_moves_tabs_between_windows() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        st.workspace.set_node(layout::parse("H(0.70,3D,V(0.50,[Graph*,Orbitals],[Log*]))").unwrap());
        let outs = tab_menu(&ctx, &mut st, &inp, DockTab::Graph, "tab-menu-new-window");
        assert!(outs.iter().all(|o| o.actions.is_empty() && o.commands.is_empty()), "display state");
        let id = match st.workspace.windows().as_slice() {
            [id] => *id,
            w => panic!("{w:?}"),
        };
        assert_eq!(st.workspace.window_panels(id), vec![DockTab::Graph]);
        assert!(matches!(requests(&outs).as_slice(), [WindowRequest::Open { panels, .. }] if *panels == vec![DockTab::Graph]), "{:?}", requests(&outs));
        assert_eq!(layout::encode(&st.workspace.to_node()), "H(0.70,3D,V(0.50,[Orbitals*],[Log*]))");
        // Orbitals into that window by its number
        tab_menu(&ctx, &mut st, &inp, DockTab::Orbitals, &format!("tab-menu-window-{id}"));
        assert_eq!(st.workspace.window_panels(id), vec![DockTab::Graph, DockTab::Orbitals]);
        assert_eq!(layout::encode(&st.workspace.to_node()), "H(0.70,3D,[Log*])");
        // from inside the window, Orbitals back to the main window: it left a leaf of its own, so
        // beside the 3D view on the right
        let wctx = egui::Context::default();
        apply_theme(&wctx);
        PROBES.with(|m| m.borrow_mut().clear());
        let outs = tab_menu_in_window(&wctx, &mut st, &inp, id, DockTab::Orbitals, "tab-menu-main");
        assert!(outs.iter().all(|o| o.actions.is_empty() && o.commands.is_empty()), "display state");
        assert_eq!(st.workspace.window_panels(id), vec![DockTab::Graph]);
        assert_eq!(layout::encode(&st.workspace.to_node()), "H(0.70,H(0.70,3D,[Orbitals*]),[Log*])");
    }

    /// Review I2: the tab menu's Reset the layout with two monitors (it runs while the main tree
    /// is lent out) applies Two monitors whole: one detached window holding Activity over Graph,
    /// both in the workspace — from the default layout, and again from Two monitors itself (the
    /// old window closes, a new one opens).
    #[test]
    fn reset_from_the_tab_menu_with_two_monitors_opens_the_window() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        st.workspace.set_monitors(vec![PRIMARY, SECOND]);
        let check = |st: &UiState, outs: &[PanelOutput]| -> u32 {
            assert!(outs.iter().all(|o| o.actions.is_empty() && o.commands.is_empty()), "display state");
            let id = match st.workspace.windows().as_slice() {
                [id] => *id,
                w => panic!("one window: {w:?}"),
            };
            assert_eq!(st.workspace.window_panels(id), vec![DockTab::Activity, DockTab::Graph]);
            assert!(st.workspace.present(DockTab::Activity) && st.workspace.present(DockTab::Graph), "neither closed");
            assert_eq!(st.workspace.to_node(), Preset::TwoMonitors.tree().0);
            let open = WindowRequest::Open { id, panels: vec![DockTab::Activity, DockTab::Graph], rect: Some(SECOND), size: None, fill: true };
            assert!(requests(outs).contains(&open), "{:?}", requests(outs));
            id
        };
        let outs = tab_menu(&ctx, &mut st, &inp, DockTab::Log, "tab-menu-reset");
        let first = check(&st, &outs);
        let outs = tab_menu(&ctx, &mut st, &inp, DockTab::Log, "tab-menu-reset");
        let second = check(&st, &outs);
        assert_ne!(first, second);
        assert!(requests(&outs).contains(&WindowRequest::Close { id: first }), "{:?}", requests(&outs));
    }

    /// The leaf's pop-out button (between its pin and its close) moves the leaf's panels into a
    /// new window and asks the window side to open it; the window draws them in its own context
    /// (their tabs and bodies, no pin); its Dock back button returns them to the main window and
    /// asks for the window to close. None of it is an action.
    #[test]
    fn the_pop_out_button_detaches_a_leaf_and_dock_back_returns_it() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        st.workspace.set_node(layout::parse("H(0.70,3D,V(0.50,[Graph*,Orbitals],[Log*]))").unwrap());
        let outs = click_widget(&ctx, &mut st, &inp, "dock-pop-Graph");
        assert!(outs.iter().all(|o| o.actions.is_empty() && o.commands.is_empty()), "display state");
        assert_eq!(layout::encode(&st.workspace.to_node()), "H(0.70,3D,[Log*])");
        let ids = st.workspace.windows();
        assert_eq!(ids.len(), 1);
        let id = ids[0];
        assert_eq!(layout::encode(&st.workspace.window_node(id).unwrap()), "[Graph*,Orbitals]");
        match requests(&outs).as_slice() {
            [WindowRequest::Open { id: o, panels, rect: None, size: Some(_), fill: false }] => assert_eq!((*o, panels.clone()), (id, vec![DockTab::Graph, DockTab::Orbitals])),
            r => panic!("{r:?}"),
        }
        assert!(st.tab_shown(DockTab::Graph) && st.workspace.present(DockTab::Orbitals) && !st.tab_shown(DockTab::Orbitals));
        // the window's own context draws its tabs (and no pin: it has no strips)
        let wctx = egui::Context::default();
        apply_theme(&wctx);
        PROBES.with(|m| m.borrow_mut().clear());
        for _ in 0..2 {
            assert!(window_frame(&wctx, &mut st, &inp, id, vec![]).actions.is_empty());
        }
        let (tab, pop, close) = (probed_rect("dock-tab-Graph"), probed_rect("dock-pop-Graph"), probed_rect("dock-close-Graph"));
        assert!(tab.right() < pop.left() && (pop.right() - close.left()).abs() < 0.5 && close.right() <= 600.5, "tab {tab:?} pop {pop:?} close {close:?}");
        assert!(PROBES.with(|m| !m.borrow().contains_key("dock-pin-Graph")), "no pin in a detached window");
        // Dock back: both panels into the main window — their leaf is gone, so together beside the
        // 3D view on the right, the shown one still shown — and the window closes
        let outs = click_in_window(&wctx, &mut st, &inp, id, "dock-pop-Graph");
        assert!(outs.iter().all(|o| o.actions.is_empty() && o.commands.is_empty()), "display state");
        assert!(st.workspace.windows().is_empty());
        assert_eq!(requests(&outs), vec![WindowRequest::Close { id }]);
        assert_eq!(layout::encode(&st.workspace.to_node()), "H(0.70,H(0.70,3D,[Graph*,Orbitals]),[Log*])");
    }

    /// A panel's body draws in a detached window's own context, from the same state, and counts
    /// as shown there.
    #[test]
    fn a_detached_panel_draws_in_its_window() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        let id = st.workspace.pop_out(&[DockTab::Selection], DockTab::Selection).unwrap();
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        assert!(!layout::panels(&st.workspace.to_node()).contains(&DockTab::Selection));
        let wctx = egui::Context::default();
        apply_theme(&wctx);
        PROBES.with(|m| m.borrow_mut().clear());
        window_frame(&wctx, &mut st, &inp, id, vec![]);
        window_frame(&wctx, &mut st, &inp, id, vec![]);
        assert!(PROBES.with(|m| m.borrow().contains_key("dock-tab-Selection")));
        assert!(st.tab_shown(DockTab::Selection));
        assert!(!window_frame(&wctx, &mut st, &inp, id + 7, vec![]).windows.iter().any(|r| matches!(r, WindowRequest::Open { .. })), "an unknown window draws nothing");
    }

    /// A panel's widget clicked in a detached window: its action comes out of that window's
    /// frame (the viewer appends it to the main frame's, so `--record` sees it).
    #[test]
    fn a_detached_panels_actions_come_out_of_its_window() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        let id = st.workspace.pop_out(&[DockTab::View], DockTab::View).unwrap();
        st.view_sections = dock::view::Sections::all(false);
        st.view_sections.set(dock::view::Section::Bonds, true);
        let was = st.show_bonds;
        let wctx = egui::Context::default();
        apply_theme(&wctx);
        let outs = click_in_window(&wctx, &mut st, &inp, id, "show-bonds");
        let acts: Vec<&Action> = outs.iter().flat_map(|o| &o.actions).collect();
        assert!(matches!(acts.as_slice(), [Action::ShowBonds(b)] if *b == !was), "{acts:?}");
        assert_eq!(st.show_bonds, !was, "applied as the main window's are");
    }

    /// Closing the OS window (the window side reports it) docks its panels back where they
    /// left: Activity into the leaf it shared with Log and Graph.
    #[test]
    fn closing_the_window_returns_activity_to_its_old_leaf() {
        let scene = Scene::preset("h2").unwrap();
        let mut st = UiState::new("h2", &scene);
        let before = layout::encode(&st.workspace.to_node());
        let id = st.workspace.pop_out(&[DockTab::Activity], DockTab::Activity).unwrap();
        assert!(!layout::panels(&st.workspace.to_node()).contains(&DockTab::Activity));
        assert_eq!(st.workspace.window_panels(id), vec![DockTab::Activity]);
        st.workspace.window_closed(id);
        assert!(st.workspace.windows().is_empty());
        assert_eq!(layout::encode(&st.workspace.to_node()), before.replace("[Log*,Activity,Graph]", "[Log,Graph,Activity*]"));
    }

    /// The profile writes one `window = x,y,w,h <tree>` line per detached window and reads them
    /// back: the same windows, rects and trees, each panel once; loaded with only the main
    /// monitor connected, the window on the second monitor docks back.
    #[test]
    fn the_profile_keeps_the_detached_windows() {
        let scene = Scene::preset("h2").unwrap();
        let mut st = UiState::new("h2", &scene);
        assert!(!profile::capture(&st).contains("window ="), "none by default");
        let a = st.workspace.pop_out(&[DockTab::Activity, DockTab::Graph], DockTab::Graph).unwrap();
        st.workspace.set_window_rect(a, Some(WinRect { x: 2600, y: 30, w: 900, h: 1000 }), true);
        st.workspace.pop_out(&[DockTab::Log], DockTab::Log).unwrap();
        let text = profile::capture(&st);
        assert!(text.contains("\nwindow = 2600,30,900,1000 max [Activity,Graph*]\nwindow = [Log*]\n"), "{text}");
        assert!(text.contains(&format!("\nlayout = {}\n", layout::encode(&st.workspace.to_node()))), "{text}");
        let mut back = UiState::new("h2", &scene);
        profile::apply(&mut back, &text);
        assert_eq!(layout::encode(&back.workspace.to_node()), layout::encode(&st.workspace.to_node()));
        assert_eq!(back.workspace.windows().len(), 2);
        assert_eq!(back.workspace.saved_windows(), st.workspace.saved_windows());
        assert_eq!(profile::capture(&back), text, "round trip");
        // a profile with a layout and no window lines has no detached windows
        profile::apply(&mut back, &format!("layout = {}\n", layout::encode(&layout::default_layout())));
        assert!(back.workspace.windows().is_empty());
        // loaded with only the main monitor connected: the window on the second docks back, the
        // one without a rect opens
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut one = UiState::new("h2", &scene);
        one.workspace.set_monitors(vec![PRIMARY]);
        profile::apply(&mut one, &text);
        let out = frame(&ctx, &mut one, &inp, 1400.0, vec![]);
        let opened: Vec<Vec<DockTab>> = out.windows.iter().filter_map(|r| if let WindowRequest::Open { panels, .. } = r { Some(panels.clone()) } else { None }).collect();
        assert_eq!(opened, vec![vec![DockTab::Log]]);
        let main = layout::panels(&one.workspace.to_node());
        assert!(main.contains(&DockTab::Activity) && main.contains(&DockTab::Graph));
        assert!(one.status.contains("no connected monitor"), "{}", one.status);
    }

    /// Two monitors (spec D-11): with a second monitor, Activity over Graph in a window filling
    /// it (maximised there) and the rest as One monitor less those two; Reset layout is then Two
    /// monitors too. With one monitor it is One monitor and the status says so.
    #[test]
    fn two_monitors_fills_the_second_monitor_or_says_there_is_one() {
        let scene = Scene::preset("h2").unwrap();
        let mut st = UiState::new("h2", &scene);
        st.workspace.set_monitors(vec![PRIMARY, SECOND]);
        st.apply_layout(Preset::TwoMonitors);
        assert_eq!(st.workspace.to_node(), Preset::TwoMonitors.tree().0);
        let ids = st.workspace.windows();
        assert_eq!(ids.len(), 1);
        assert_eq!(layout::encode(&st.workspace.window_node(ids[0]).unwrap()), "V(0.50,[Activity*],[Graph*])");
        let open = |id: u32| WindowRequest::Open { id, panels: vec![DockTab::Activity, DockTab::Graph], rect: Some(SECOND), size: None, fill: true };
        assert_eq!(st.workspace.take_requests(), vec![open(ids[0])]);
        // applying it again replaces the window (the old one closes)
        st.apply_layout(Preset::TwoMonitors);
        assert_eq!(st.workspace.take_requests(), vec![open(ids[0] + 1), WindowRequest::Close { id: ids[0] }]);
        // One monitor closes it, every panel back in the main tree
        st.apply_layout(Preset::OneMonitor);
        assert!(st.workspace.windows().is_empty() && st.workspace.to_node() == layout::default_layout());
        st.reset_layout();
        assert_eq!(st.workspace.windows().len(), 1, "Reset layout with a second monitor: Two monitors");
        // one monitor: One monitor, with a note
        let mut one = UiState::new("h2", &scene);
        one.workspace.set_monitors(vec![PRIMARY]);
        one.apply_layout(Preset::TwoMonitors);
        assert!(one.workspace.windows().is_empty() && one.workspace.to_node() == layout::default_layout());
        assert!(one.status.starts_with("Only one monitor found"), "{}", one.status);
        one.reset_layout();
        assert!(one.workspace.windows().is_empty());
    }

    /// Ctrl K revealing a control of a panel in a detached window shows the panel there and asks
    /// for the window to come to the front (spec D-12).
    #[test]
    fn ctrl_k_reveal_focuses_the_detached_window() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        let id = st.workspace.pop_out(&[DockTab::Graph, DockTab::Activity], DockTab::Graph).unwrap();
        let first = frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        assert!(matches!(first.windows.as_slice(), [WindowRequest::Open { .. }]));
        assert!(!st.tab_shown(DockTab::Activity));
        let entry = search::ENTRIES.iter().find(|e| e.place == search::Place::Dock(DockTab::Activity)).expect("an Activity control");
        search::pick(&ctx, &mut st, entry);
        let out = frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        assert_eq!(out.windows, vec![WindowRequest::Focus { id }]);
        assert!(st.tab_shown(DockTab::Activity) && st.workspace.window_of(DockTab::Activity) == Some(id), "shown in its window, not moved");
    }

    // ---- dragging tabs between windows (docking spec D-15, D-5; user decision 2026-11-12) ----

    /// Window `win` (`None`: the main one) on the desktop: its client area at `(x, y)`, `w × h`
    /// points at `ppp` pixels per point, its native frame round it (8 px borders, a 31 px title
    /// bar; the main window's custom frame has none).
    fn place(win: Option<u32>, x: i32, y: i32, w: f32, h: f32, ppp: f32) -> Place {
        let inner = WinRect { x, y, w: (w * ppp) as u32, h: (h * ppp) as u32 };
        let outer = match win {
            None => inner,
            Some(_) => WinRect { x: x - 8, y: y - 31, w: inner.w + 16, h: inner.h + 39 },
        };
        Place { win, inner, outer, targets: true }
    }

    /// [`window_frame`] at `ppp` pixels per point (its 600 × 800 points then `600 ppp × 800 ppp`
    /// pixels: a window on a monitor of another scale).
    fn window_frame_at(ctx: &egui::Context, st: &mut UiState, inp: &PanelInput<'_>, id: u32, ppp: f32, events: Vec<egui::Event>) -> PanelOutput {
        let mut raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(600.0, 800.0))), events, ..Default::default() };
        raw.viewports.entry(egui::ViewportId::ROOT).or_default().native_pixels_per_point = Some(ppp);
        let mut out = None;
        let mut full = ctx.run_ui(raw, |root| out = Some(show_window(root, st, inp, id)));
        full.textures_delta.clear();
        out.unwrap()
    }

    /// Drags tab `t` in one window (`src` draws that window's frame with the events given) and
    /// releases it at the end of `path` (the source window's points, past its edges too: a held
    /// button's pointer goes on being reported to the window it was pressed in), `esc` pressed
    /// on the way when asked. The other windows (`others`) draw after each of the source's frames,
    /// as every window draws in each of the main window's frames. Returns every frame's output
    /// and the drop target the others drew just before the release (`None`: none).
    #[allow(clippy::type_complexity)]
    fn drag_tab(st: &mut UiState, t: DockTab, path: &[egui::Pos2], esc: bool, src: &mut dyn FnMut(&mut UiState, Vec<egui::Event>) -> PanelOutput, others: &mut dyn FnMut(&mut UiState) -> Vec<PanelOutput>) -> (Vec<PanelOutput>, Option<egui::Rect>) {
        for _ in 0..3 {
            src(st, vec![]);
            others(st);
        }
        let c = probed_rect(&format!("dock-tab-{t:?}")).center();
        let mut outs = vec![src(st, vec![egui::Event::PointerMoved(c), pointer(c, true)])];
        outs.extend(others(st));
        // decidedly dragging (egui), then past egui_dock's 30-pt threshold, then the path
        let mut steps = vec![c + egui::vec2(12.0, 0.0), c + egui::vec2(50.0, 0.0)];
        steps.extend_from_slice(path);
        let end = *steps.last().unwrap();
        for (i, &p) in steps.iter().enumerate() {
            let mut ev = vec![egui::Event::PointerMoved(p)];
            if esc && i == 2 {
                ev.push(key(egui::Key::Escape, egui::Modifiers::NONE));
            }
            outs.push(src(st, ev));
            if i == 1 {
                assert_eq!(st.workspace.dragging(), Some(t), "the drag is latched from its tab button");
            }
            outs.extend(others(st));
        }
        PROBES.with(|m| m.borrow_mut().remove("drop-target"));
        // held still a moment at the end (egui_dock keeps its first target for 0.3 s; a frame is
        // 1/60 s headless)
        for _ in 0..24 {
            outs.push(src(st, vec![egui::Event::PointerMoved(end)]));
            outs.extend(others(st));
        }
        let drawn = PROBES.with(|m| m.borrow().get("drop-target").copied());
        outs.push(src(st, vec![pointer(end, false)]));
        outs.extend(others(st));
        outs.push(src(st, vec![egui::Event::PointerGone]));
        outs.extend(others(st));
        assert_eq!(st.workspace.dragging(), None, "the drag is over");
        (outs, drawn)
    }

    fn opened(outs: &[PanelOutput]) -> Vec<(u32, Vec<DockTab>, Option<WinRect>)> {
        requests(outs)
            .into_iter()
            .filter_map(|r| match r {
                WindowRequest::Open { id, panels, rect, .. } => Some((id, panels, rect)),
                _ => None,
            })
            .collect()
    }

    /// A tab dragged out of the main window and released over none of the app's windows opens
    /// one new window holding it (spec D-15): its top-left at the release point, the panel's leaf
    /// size, kept inside the work area of the monitor under the cursor (here pushed up off the
    /// taskbar). The tab leaves the main tree; nothing floats; none of it is an action. Esc on
    /// the way cancels.
    #[test]
    fn a_tab_dragged_out_of_every_window_opens_one_at_the_release_point() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        st.workspace.set_node(layout::parse("H(0.70,3D,V(0.50,[Graph*,Orbitals],[Log*]))").unwrap());
        st.workspace.set_monitors(vec![PRIMARY, SECOND]);
        st.workspace.set_work_areas(vec![WinRect { x: 0, y: 0, w: 2560, h: 1400 }, SECOND]);
        st.workspace.set_places(vec![place(None, 0, 0, 1400.0, 900.0, 1.0)]);
        for _ in 0..3 {
            frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        }
        let leaf = st.workspace.leaf_rect(DockTab::Graph).unwrap();
        let (w, h) = (leaf.width().round() as u32, leaf.height().round() as u32);
        let mut main = |st: &mut UiState, ev: Vec<egui::Event>| frame(&ctx, st, &inp, 1400.0, ev);
        let (outs, drawn) = drag_tab(&mut st, DockTab::Graph, &[egui::pos2(1700.0, 1300.0)], false, &mut main, &mut |_| vec![]);
        assert!(outs.iter().all(|o| o.actions.is_empty() && o.commands.is_empty()), "display state");
        assert_eq!(drawn, None, "no window under the cursor draws a target");
        let id = st.workspace.window_of(DockTab::Graph).expect("in a new window");
        assert_eq!(opened(&outs), vec![(id, vec![DockTab::Graph], Some(WinRect { x: 1700, y: 1400 - h as i32, w, h }))]);
        assert_eq!(layout::encode(&st.workspace.to_node()), "H(0.70,3D,V(0.50,[Orbitals*],[Log*]))");
        assert_eq!(st.workspace.floats_for_test(), 0);
        // on the second monitor, beyond its right edge less the window's width: kept on it
        let (outs, _) = drag_tab(&mut st, DockTab::Orbitals, &[egui::pos2(4400.0, 50.0)], false, &mut main, &mut |_| vec![]);
        let r = opened(&outs)[0].2.unwrap();
        assert!(r.x + r.w as i32 == 2560 + 1920 && r.y == 50, "{r:?}");
        // Esc on the way: the release does nothing, and the Esc is the drag's alone — it does not
        // also deselect (Task 11 review M4)
        let (outs, _) = drag_tab(&mut st, DockTab::Log, &[egui::pos2(1700.0, 500.0)], true, &mut main, &mut |_| vec![]);
        assert!(opened(&outs).is_empty() && st.workspace.window_of(DockTab::Log).is_none());
        assert!(outs.iter().all(|o| !o.deselect && o.actions.is_empty()), "Esc was not also a Deselect");
        assert_eq!(layout::encode(&st.workspace.to_node()), "H(0.70,3D,[Log*])");
        // released over the capture pop-out view (one of the app's windows, no drop targets):
        // nothing opens, the tab stays (review M2)
        st.workspace.set_places(vec![popout_place(1600, 1000, 800, 400), place(None, 0, 0, 1400.0, 900.0, 1.0)]);
        let (outs, _) = drag_tab(&mut st, DockTab::Log, &[egui::pos2(1700.0, 1100.0)], false, &mut main, &mut |_| vec![]);
        assert!(opened(&outs).is_empty() && st.workspace.window_of(DockTab::Log).is_none());
        assert_eq!(layout::encode(&st.workspace.to_node()), "H(0.70,3D,[Log*])");
    }

    /// A tab dragged from the main window over a detached window: that window draws the drop
    /// targets at the cursor, and the release on its leaf's centre tabs the panel in, on an edge
    /// button splits, on its tab bar inserts before the tab under the pointer (spec D-15). The
    /// detached window is at 2 pixels per point and the main one at 1 (mixed DPI: each converts
    /// with its own).
    #[test]
    fn a_drag_over_a_detached_window_docks_the_tab_there() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let (ctx, wctx) = (egui::Context::default(), egui::Context::default());
        apply_theme(&ctx);
        apply_theme(&wctx);
        let mut st = UiState::new("h2", &scene);
        st.workspace.set_node(layout::parse("H(0.70,3D,V(0.50,[Graph*,Orbitals],[Log*,Model,Activity]))").unwrap());
        let id = st.workspace.pop_out(&[DockTab::Activity], DockTab::Activity).unwrap();
        st.workspace.take_requests();
        st.workspace.set_places(vec![place(Some(id), 1500, 100, 600.0, 800.0, 2.0), place(None, 0, 0, 1400.0, 900.0, 1.0)]);
        // a point of the detached window (its points) as the main window sees it (its points)
        let at = |x: f32, y: f32| egui::pos2(1500.0 + 2.0 * x, 100.0 + 2.0 * y);
        let mut main = |st: &mut UiState, ev: Vec<egui::Event>| frame(&ctx, st, &inp, 1400.0, ev);
        let mut win = |st: &mut UiState| vec![window_frame_at(&wctx, st, &inp, id, 2.0, vec![])];
        // its leaf's centre (the leaf fills its 600 × 800 points)
        let (outs, drawn) = drag_tab(&mut st, DockTab::Log, &[at(300.0, 400.0)], false, &mut main, &mut win);
        assert!(outs.iter().all(|o| o.actions.is_empty() && o.commands.is_empty()), "display state");
        assert!(drawn.is_some_and(|r| r.contains(egui::pos2(300.0, 400.0))), "the centre button is lit: {drawn:?}");
        assert_eq!(layout::encode(&st.workspace.window_node(id).unwrap()), "[Activity,Log*]");
        assert!(opened(&outs).is_empty() && st.workspace.windows() == vec![id]);
        // the left edge button: a 100-pt button 10 pt left of the centre one
        let (_, drawn) = drag_tab(&mut st, DockTab::Model, &[at(190.0, 400.0)], false, &mut main, &mut win);
        assert!(drawn.is_some_and(|r| r.contains(egui::pos2(190.0, 400.0))), "{drawn:?}");
        assert_eq!(layout::encode(&st.workspace.window_node(id).unwrap()), "H(0.50,[Model*],[Activity,Log*])");
        // its tab bar, on the Activity tab: Graph goes before it
        for _ in 0..2 {
            win(&mut st);
        }
        let tab = probed_rect("dock-tab-Activity").center();
        let (_, drawn) = drag_tab(&mut st, DockTab::Graph, &[at(tab.x, tab.y)], false, &mut main, &mut win);
        assert!(drawn.is_some_and(|r| r.contains(tab)), "the tab is lit: {drawn:?}");
        assert_eq!(layout::encode(&st.workspace.window_node(id).unwrap()), "H(0.50,[Model*],[Graph*,Activity,Log])");
        assert_eq!(layout::encode(&st.workspace.to_node()), "H(0.70,3D,[Orbitals*])");
        assert_eq!(st.workspace.floats_for_test(), 0);
        // over the window's title bar (its frame, not its client area): no target, nothing moves
        let (_, drawn) = drag_tab(&mut st, DockTab::Orbitals, &[egui::pos2(1800.0, 80.0)], false, &mut main, &mut win);
        assert_eq!(drawn, None);
        assert_eq!(layout::encode(&st.workspace.to_node()), "H(0.70,3D,[Orbitals*])");
        assert!(st.workspace.windows() == vec![id]);
    }

    /// A tab dragged from a detached window into the main window: its 3D view's tile takes edge
    /// splits only (spec D-3: the centre is refused, the tab stays), its left edge button splits
    /// the panel in beside the view; the window, left empty, closes.
    #[test]
    fn a_drag_from_a_detached_window_docks_into_the_main_window_and_the_empty_window_closes() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let (ctx, wctx) = (egui::Context::default(), egui::Context::default());
        apply_theme(&ctx);
        apply_theme(&wctx);
        let mut st = UiState::new("h2", &scene);
        st.workspace.set_node(layout::parse("H(0.70,3D,[Log*,Graph])").unwrap());
        let id = st.workspace.pop_out(&[DockTab::Graph], DockTab::Graph).unwrap();
        st.workspace.take_requests();
        st.workspace.set_places(vec![place(Some(id), 1500, 100, 600.0, 800.0, 1.0), place(None, 0, 0, 1400.0, 900.0, 1.0)]);
        let view = frame(&ctx, &mut st, &inp, 1400.0, vec![]).view.unwrap();
        // a point of the main window (its points) as the detached one sees it
        let at = |p: egui::Pos2| egui::pos2(p.x - 1500.0, p.y - 100.0);
        let mut win = |st: &mut UiState, ev: Vec<egui::Event>| window_frame_at(&wctx, st, &inp, id, 1.0, ev);
        let mut main = |st: &mut UiState| vec![frame(&ctx, st, &inp, 1400.0, vec![])];
        let (_, drawn) = drag_tab(&mut st, DockTab::Graph, &[at(view.center())], false, &mut win, &mut main);
        assert_eq!(drawn, None, "the 3D view's tile has no centre target");
        assert_eq!(st.workspace.window_panels(id), vec![DockTab::Graph], "refused: it stays");
        let left = view.center() - egui::vec2(110.0, 0.0);
        let (outs, drawn) = drag_tab(&mut st, DockTab::Graph, &[at(left)], false, &mut win, &mut main);
        assert!(outs.iter().all(|o| o.actions.is_empty() && o.commands.is_empty()), "display state");
        assert!(drawn.is_some_and(|r| r.contains(left)), "{drawn:?}");
        assert_eq!(layout::encode(&st.workspace.to_node()), "H(0.70,H(0.50,[Graph*],3D),[Log*])");
        assert!(st.workspace.windows().is_empty());
        assert!(requests(&outs).contains(&WindowRequest::Close { id }), "{:?}", requests(&outs));
    }

    /// Detached window → detached window: the tab moves, and the window it left, emptied,
    /// closes.
    #[test]
    fn a_drag_between_detached_windows_moves_the_tab() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let (ctx, actx, bctx) = (egui::Context::default(), egui::Context::default(), egui::Context::default());
        for c in [&ctx, &actx, &bctx] {
            apply_theme(c);
        }
        let mut st = UiState::new("h2", &scene);
        let a = st.workspace.pop_out(&[DockTab::Graph], DockTab::Graph).unwrap();
        let b = st.workspace.pop_out(&[DockTab::Log], DockTab::Log).unwrap();
        st.workspace.take_requests();
        st.workspace.set_places(vec![place(Some(b), 2200, 100, 600.0, 800.0, 1.0), place(Some(a), 1500, 100, 600.0, 800.0, 1.0), place(None, 0, 0, 1400.0, 900.0, 1.0)]);
        let mut src = |st: &mut UiState, ev: Vec<egui::Event>| window_frame_at(&bctx, st, &inp, b, 1.0, ev);
        let mut others = |st: &mut UiState| vec![frame(&ctx, st, &inp, 1400.0, vec![]), window_frame_at(&actx, st, &inp, a, 1.0, vec![])];
        // window a's leaf centre, from window b's points
        let (outs, drawn) = drag_tab(&mut st, DockTab::Log, &[egui::pos2(1800.0 - 2200.0, 500.0 - 100.0)], false, &mut src, &mut others);
        assert!(drawn.is_some());
        assert_eq!(layout::encode(&st.workspace.window_node(a).unwrap()), "[Graph,Log*]");
        assert_eq!(st.workspace.windows(), vec![a]);
        assert!(requests(&outs).contains(&WindowRequest::Close { id: b }));
    }

    /// No drag-release inside a window floats a panel (spec D-5): released on a leaf's body off
    /// its drop buttons, or on the 3D view's corner, nothing moves; on a leaf's centre button
    /// `egui_dock`'s own drop still tabs it in. No window opens.
    #[test]
    fn no_drag_released_inside_a_window_floats_a_panel() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let mut st = UiState::new("h2", &scene);
        let tree = "H(0.70,3D,V(0.50,[Graph*,Orbitals],[Log*]))";
        st.workspace.set_node(layout::parse(tree).unwrap());
        st.workspace.set_places(vec![place(None, 0, 0, 1400.0, 900.0, 1.0)]);
        for _ in 0..3 {
            frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        }
        let (log, view) = (st.workspace.leaf_rect(DockTab::Log).unwrap(), frame(&ctx, &mut st, &inp, 1400.0, vec![]).view.unwrap());
        let mut main = |st: &mut UiState, ev: Vec<egui::Event>| frame(&ctx, st, &inp, 1400.0, ev);
        for spot in [log.left_bottom() + egui::vec2(12.0, -12.0), view.left_top() + egui::vec2(20.0, 20.0)] {
            let (outs, _) = drag_tab(&mut st, DockTab::Graph, &[spot], false, &mut main, &mut |_| vec![]);
            assert!(opened(&outs).is_empty(), "{spot:?}");
            assert_eq!(layout::encode(&st.workspace.to_node()), layout::encode(&layout::parse(tree).unwrap()), "{spot:?}");
            assert_eq!(st.workspace.floats_for_test(), 0, "{spot:?}");
        }
        let (outs, _) = drag_tab(&mut st, DockTab::Graph, &[log.center()], false, &mut main, &mut |_| vec![]);
        assert!(opened(&outs).is_empty());
        assert_eq!(layout::encode(&st.workspace.to_node()), "H(0.70,3D,V(0.50,[Orbitals*],[Log,Graph*]))");
        assert_eq!(st.workspace.floats_for_test(), 0);
    }

    /// The capture pop-out view on the desktop: one of the app's windows, with no drop targets.
    fn popout_place(x: i32, y: i32, w: u32, h: u32) -> Place {
        let r = WinRect { x, y, w, h };
        Place { win: None, inner: r, outer: r, targets: false }
    }

    /// Task 11 review I1: with app windows overlapping, the window behind never takes a drop its
    /// `egui_dock` resolved under the one in front. Window B covers the main window's Log leaf; a
    /// tab released on B off its targets, right over the main window's Log centre button, changes
    /// neither tree (no target: the tab stays, spec D-15). The reverse — the source window behind
    /// the main one, released on the main window off its targets over its own other leaf's centre
    /// — changes neither. A release on B's target over the main window's Log centre moves the tab
    /// once, into B.
    #[test]
    fn a_release_over_a_covering_window_never_drops_into_the_window_behind() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let (ctx, wctx) = (egui::Context::default(), egui::Context::default());
        apply_theme(&ctx);
        apply_theme(&wctx);
        let mut st = UiState::new("h2", &scene);
        let tree = "H(0.70,3D,V(0.50,[Graph*,Orbitals],[Log*]))";
        st.workspace.set_node(layout::parse("H(0.70,3D,V(0.50,[Graph*,Orbitals],[Log*,Activity]))").unwrap());
        let b = st.workspace.pop_out(&[DockTab::Activity], DockTab::Activity).unwrap();
        st.workspace.take_requests();
        assert_eq!(layout::encode(&st.workspace.to_node()), tree);
        for _ in 0..3 {
            frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        }
        let log = st.workspace.leaf_rect(DockTab::Log).unwrap().center();
        // B's point (40, 760), the corner of its body, over the main window's Log centre
        let cover = |st: &mut UiState, bx: f32, by: f32| {
            st.workspace.set_places(vec![place(Some(b), (log.x - bx) as i32, (log.y - by) as i32, 600.0, 800.0, 1.0), place(None, 0, 0, 1400.0, 900.0, 1.0)]);
        };
        cover(&mut st, 40.0, 760.0);
        let mut main = |st: &mut UiState, ev: Vec<egui::Event>| frame(&ctx, st, &inp, 1400.0, ev);
        let mut win = |st: &mut UiState| vec![window_frame_at(&wctx, st, &inp, b, 1.0, vec![])];
        let (outs, drawn) = drag_tab(&mut st, DockTab::Graph, &[log], false, &mut main, &mut win);
        assert_eq!(drawn, None, "B's corner is no target");
        assert!(opened(&outs).is_empty());
        assert_eq!(layout::encode(&st.workspace.to_node()), tree, "the main window's hidden drop was undone");
        assert_eq!(layout::encode(&st.workspace.window_node(b).unwrap()), "[Activity*]");
        // B's centre over the main window's Log centre: into B, once
        cover(&mut st, 300.0, 400.0);
        let (_, drawn) = drag_tab(&mut st, DockTab::Graph, &[log], false, &mut main, &mut win);
        assert!(drawn.is_some());
        assert_eq!(layout::encode(&st.workspace.window_node(b).unwrap()), "[Activity,Graph*]");
        assert_eq!(layout::encode(&st.workspace.to_node()), "H(0.70,3D,V(0.50,[Orbitals*],[Log*]))");
        assert_eq!(st.workspace.floats_for_test(), 0);

        // the reverse: window C (two leaves) behind the main window, its Model leaf's centre under
        // the main window's Log leaf, off that leaf's buttons
        let mut st = UiState::new("h2", &scene);
        st.workspace.set_node(layout::parse(tree).unwrap());
        st.workspace.set_windows(vec![(None, false, layout::parse_window("V(0.50,[Activity*],[Model*])").unwrap())]);
        let c = st.workspace.windows()[0];
        st.workspace.take_requests();
        for _ in 0..3 {
            frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        }
        let spot = st.workspace.leaf_rect(DockTab::Log).unwrap().left_bottom() + egui::vec2(12.0, -12.0);
        let (cx, cy) = ((spot.x - 300.0) as i32, (spot.y - 600.0) as i32);
        st.workspace.set_places(vec![place(None, 0, 0, 1400.0, 900.0, 1.0), place(Some(c), cx, cy, 600.0, 800.0, 1.0)]);
        let cctx = egui::Context::default();
        apply_theme(&cctx);
        let mut src = |st: &mut UiState, ev: Vec<egui::Event>| window_frame_at(&cctx, st, &inp, c, 1.0, ev);
        let mut others = |st: &mut UiState| vec![frame(&ctx, st, &inp, 1400.0, vec![])];
        let (outs, drawn) = drag_tab(&mut st, DockTab::Activity, &[egui::pos2(300.0, 600.0)], false, &mut src, &mut others);
        assert_eq!(drawn, None, "the main window's Log corner is no target");
        assert!(opened(&outs).is_empty());
        assert_eq!(layout::encode(&st.workspace.window_node(c).unwrap()), "V(0.50,[Activity*],[Model*])", "C's hidden drop was undone");
        assert_eq!(layout::encode(&st.workspace.to_node()), tree);
    }

    /// Task 11 review M7: tearing out a detached window's only tab closes that window and opens a
    /// new one holding the tab where it was released (its leaf's size, kept on the second
    /// monitor).
    #[test]
    fn tearing_out_a_detached_windows_only_tab_reopens_it_where_released() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let (ctx, wctx) = (egui::Context::default(), egui::Context::default());
        apply_theme(&ctx);
        apply_theme(&wctx);
        let mut st = UiState::new("h2", &scene);
        st.workspace.set_node(layout::parse("H(0.70,3D,[Log*,Graph])").unwrap());
        let id = st.workspace.pop_out(&[DockTab::Graph], DockTab::Graph).unwrap();
        st.workspace.take_requests();
        st.workspace.set_monitors(vec![PRIMARY, SECOND]);
        st.workspace.set_places(vec![place(Some(id), 1500, 100, 600.0, 800.0, 1.0), place(None, 0, 0, 1400.0, 900.0, 1.0)]);
        let mut win = |st: &mut UiState, ev: Vec<egui::Event>| window_frame_at(&wctx, st, &inp, id, 1.0, ev);
        let mut main = |st: &mut UiState| vec![frame(&ctx, st, &inp, 1400.0, vec![])];
        // desktop (3000, 300), on the second monitor, from the window's points
        let (outs, _) = drag_tab(&mut st, DockTab::Graph, &[egui::pos2(3000.0 - 1500.0, 300.0 - 100.0)], false, &mut win, &mut main);
        assert!(outs.iter().all(|o| o.actions.is_empty() && o.commands.is_empty()), "display state");
        let new = st.workspace.window_of(DockTab::Graph).expect("in a new window");
        assert_ne!(new, id);
        assert_eq!(st.workspace.windows(), vec![new]);
        let r = requests(&outs);
        assert_eq!(r.iter().filter(|r| **r == WindowRequest::Close { id }).count(), 1, "{r:?}");
        assert_eq!(opened(&outs), vec![(new, vec![DockTab::Graph], Some(WinRect { x: 3000, y: 1080 - 800, w: 600, h: 800 }))]);
    }

    /// Task 11 review M3: a drag whose source window closes under it is forgotten — the window
    /// it was over stops drawing its targets.
    #[test]
    fn a_drag_whose_window_closes_is_forgotten() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let (ctx, wctx) = (egui::Context::default(), egui::Context::default());
        apply_theme(&ctx);
        apply_theme(&wctx);
        let mut st = UiState::new("h2", &scene);
        st.workspace.set_node(layout::parse("H(0.70,3D,[Log*,Graph])").unwrap());
        let id = st.workspace.pop_out(&[DockTab::Graph], DockTab::Graph).unwrap();
        st.workspace.take_requests();
        st.workspace.set_places(vec![place(Some(id), 1500, 100, 600.0, 800.0, 1.0), place(None, 0, 0, 1400.0, 900.0, 1.0)]);
        for _ in 0..3 {
            window_frame_at(&wctx, &mut st, &inp, id, 1.0, vec![]);
            frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        }
        let log = st.workspace.leaf_rect(DockTab::Log).unwrap().center();
        let c = probed_rect("dock-tab-Graph").center();
        window_frame_at(&wctx, &mut st, &inp, id, 1.0, vec![egui::Event::PointerMoved(c), pointer(c, true)]);
        for p in [c + egui::vec2(12.0, 0.0), c + egui::vec2(50.0, 0.0), egui::pos2(log.x - 1500.0, log.y - 100.0)] {
            window_frame_at(&wctx, &mut st, &inp, id, 1.0, vec![egui::Event::PointerMoved(p)]);
        }
        PROBES.with(|m| m.borrow_mut().remove("drop-target"));
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        assert_eq!(st.workspace.dragging(), Some(DockTab::Graph));
        assert!(PROBES.with(|m| m.borrow().contains_key("drop-target")), "the main window draws the Log leaf's targets");
        // the window closes under the drag (Alt+F4 with the button held)
        st.workspace.window_closed(id);
        assert_eq!(st.workspace.dragging(), None);
        PROBES.with(|m| m.borrow_mut().remove("drop-target"));
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        assert!(PROBES.with(|m| !m.borrow().contains_key("drop-target")), "no stale targets");
    }

    /// Docking spec D-16 for a resize grip (the brief's "separator's / grip's"; the app has none
    /// left since floats went — every `egui::Window` of the chrome is fixed-size — so a plain
    /// resizable egui window stands in): a press 2 pt outside its drawn right edge, inside egui's
    /// resize grab, is the window's (it widens) though egui-winit's `consumed` says no;
    /// `ui::takes_press`, which gates the 3D view's presses (`Viewer::ui_takes_press`), says yes.
    #[test]
    fn a_press_in_a_resize_grips_margin_is_the_uis() {
        let ctx = egui::Context::default();
        let run = |events: Vec<egui::Event>| {
            let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1000.0, 800.0))), events, ..Default::default() };
            let mut full = ctx.run_ui(raw, |root| {
                egui::Window::new("grip").id(egui::Id::new("grip-window")).default_pos([100.0, 100.0]).default_size([300.0, 200.0]).resizable(true).show(root.ctx(), |ui| {
                    ui.label("a resizable window");
                });
            });
            full.textures_delta.clear();
        };
        for _ in 0..3 {
            run(vec![]);
        }
        let rect = || ctx.memory(|m| m.area_rect(egui::Id::new("grip-window"))).expect("the window");
        let before = rect();
        let p = egui::pos2(before.right() + 2.0, before.center().y);
        run(vec![egui::Event::PointerMoved(p)]);
        assert!(!ctx.egui_wants_pointer_input(), "egui-winit's `consumed` misses the grip's margin");
        assert!(takes_press(&ctx, p), "the hit test does not");
        run(vec![pointer(p, true)]);
        for k in 1..=4 {
            run(vec![egui::Event::PointerMoved(p + egui::vec2(10.0 * k as f32, 0.0))]);
        }
        run(vec![pointer(p + egui::vec2(40.0, 0.0), false)]);
        run(vec![]);
        assert!(rect().width() > before.width() + 20.0, "egui took the press: the window widened {before:?} → {:?}", rect());
        // well away from it, the press is the 3D view's
        assert!(!takes_press(&ctx, egui::pos2(800.0, 700.0)));
    }
}

/// Train this segment (M8a Task 10): the Selection tab's button sends `Action::TrainSegment` with the
/// selection and opens the preview (the run untouched otherwise: no scene change); the card's Open
/// saves the scene, switches to the estimated solver on the segment with the Standard plan ready and
/// shows the "back to" chip, whose click brings the saved scene back on the solver it had. The
/// Model tab has the same button; an empty selection disables it. While the feature is deferred
/// (`segment::SEGMENT_TRAINING` off) neither tab draws the button, and the same flow is driven by
/// the action itself — the preview card, Open and the chip are unchanged.
#[test]
fn the_train_segment_button_opens_the_segment_and_the_chip_goes_back() {
    let scene = crate::segment::alkane(10);
    let cam = camera();
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    st.adopt_file("decane.scene", &scene);
    st.backend = Backend::Gpu;
    st.plan.seed = 7;
    st.plan.perturbations = 3; // a custom plan: the segment opens with Standard instead
    st.open_tab(DockTab::Selection);
    let inp = PanelInput { selected: Some(0), ..input(&scene, None, &cam) };
    let outs = if crate::segment::SEGMENT_TRAINING {
        let outs = click_widget(&ctx, &mut st, &inp, "train-segment");
        let acts: Vec<&Action> = outs.iter().flat_map(|o| &o.actions).collect();
        assert_eq!(acts, vec![&Action::TrainSegment(vec![0])]);
        outs
    } else {
        PROBES.with(|p| p.borrow_mut().clear());
        for _ in 0..3 {
            frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        }
        assert!(PROBES.with(|p| !p.borrow().contains_key("train-segment")), "deferred: no button in the Selection tab");
        vec![act(&mut st, &inp, Action::TrainSegment(vec![0]))]
    };
    assert!(outs.iter().all(|o| o.error.is_none() && o.switch_scene.is_none() && o.commands.is_empty()), "a preview changes nothing");
    let seg = st.segment_preview.clone().expect("the preview");
    assert!(seg.interior[0] && !seg.caps.is_empty());
    // Open: the scene saved, the segment in the estimated solver, Standard ready, the chip
    let outs = click_widget(&ctx, &mut st, &inp, "segment-open");
    let opened = outs.iter().find_map(|o| o.switch_scene.clone()).expect("the segment opens");
    assert_eq!(opened, seg.scene);
    assert!(st.segment_preview.is_none());
    assert_eq!(st.backend, Backend::Estimated);
    assert_eq!(st.source, SceneSource::Segment("decane.scene".into()));
    assert_eq!(st.source.label(), "decane.scene · segment");
    assert_eq!(st.plan, sim::SamplingPlan { seed: 7, ..sim::SamplingPlan::default() }, "Standard, the seed kept");
    assert_eq!((st.grid_n, st.grid_h), (seg.scene.grid_n, seg.scene.grid_h));
    let back = st.segment_back.clone().expect("the way back");
    assert_eq!((back.backend, back.label.as_str(), back.caps.clone()), (Backend::Gpu, "decane.scene", seg.caps.clone()));
    let saved = Scene::parse(&std::fs::read_to_string(&back.path).unwrap()).unwrap();
    assert_eq!(saved.nuclei.len(), scene.nuclei.len());
    // the segment's view: the chip goes back
    let inp2 = input(&opened, None, &cam);
    let outs = click_widget(&ctx, &mut st, &inp2, "segment-back");
    assert!(outs.iter().flat_map(|o| &o.actions).any(|a| *a == Action::BackToScene));
    let again = outs.iter().find_map(|o| o.switch_scene.clone()).expect("the scene comes back");
    assert_eq!(again, saved);
    assert_eq!(again.nuclei.iter().map(|n| n.z).collect::<Vec<_>>(), scene.nuclei.iter().map(|n| n.z).collect::<Vec<_>>());
    assert!(again.nuclei.iter().zip(&scene.nuclei).all(|(a, b)| (0..3).all(|k| (a.pos[k] - b.pos[k]).abs() < 1e-3)));
    assert_eq!((st.backend, st.source.clone(), st.scene_path.as_str()), (Backend::Gpu, SceneSource::File("decane.scene".into()), "decane.scene"));
    assert!(st.segment_back.is_none());
    for _ in 0..3 {
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
    }
    let _ = std::fs::remove_file(&back.path);
    // the Model tab has it too; nothing selected: disabled (no action)
    st.open_tab(DockTab::Model);
    if crate::segment::SEGMENT_TRAINING {
        let outs = click_widget(&ctx, &mut st, &inp, "train-segment-model");
        assert_eq!(outs.iter().flat_map(|o| &o.actions).collect::<Vec<_>>(), vec![&Action::TrainSegment(vec![0])]);
        st.segment_preview = None;
        let none = input(&scene, None, &cam);
        let outs = click_widget(&ctx, &mut st, &none, "train-segment-model");
        assert!(outs.iter().all(|o| o.actions.is_empty()), "disabled without a selection");
    } else {
        PROBES.with(|p| p.borrow_mut().clear());
        for _ in 0..3 {
            frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        }
        assert!(PROBES.with(|p| !p.borrow().contains_key("train-segment-model")), "deferred: no button in the Model tab");
    }
    // the actions are recorded, never replayed
    for a in [Action::TrainSegment(vec![0]), Action::OpenSegment, Action::CancelSegment, Action::BackToScene] {
        assert_eq!((a.describe(), crate::replay::unreplayable_kind(&a)), (None, Some("segment")));
    }
}

/// A refused segment says why and changes nothing: an empty selection, a lone atom, the whole
/// molecule (butane-sized: twice the cutoff reaches every atom).
#[test]
fn a_refused_segment_names_the_reason_and_changes_nothing() {
    let cam = camera();
    let h2 = Scene::preset("h2").unwrap();
    let mut st = UiState::new("h2", &h2);
    let inp = input(&h2, None, &cam);
    let out = act(&mut st, &inp, Action::TrainSegment(vec![]));
    assert_eq!(out.error.as_deref(), Some(format!("Train this segment: {}", crate::segment::NOTHING_SELECTED).as_str()));
    let out = act(&mut st, &inp, Action::TrainSegment(vec![0]));
    assert!(out.error.as_deref().unwrap().contains("the whole molecule — train the scene itself"), "{:?}", out.error);
    let mut far = h2.clone();
    far.nuclei.push(crate::scene::SceneNucleus { z: 2, a: None, pos: [0.0, 9.0, 0.0], vel: [0.0; 3] });
    let inp = input(&far, None, &cam);
    let out = act(&mut st, &inp, Action::TrainSegment(vec![2]));
    assert!(out.error.as_deref().unwrap().contains("He #2 has no neighbour within 6.0 Bohr"), "{:?}", out.error);
    assert!(out.commands.is_empty() && out.switch_scene.is_none() && st.segment_preview.is_none() && st.segment_back.is_none());
}

/// Rule 0b (lone atoms answered from the atom references): its own label, colour and hover — the
/// honesty words the brief asks for (the DFT wobble, nothing learned) — distinct from the
/// surrogate's and the fallback's.
#[test]
fn a_reference_answer_has_its_own_label_colour_and_tip() {
    let r = Source::Reference;
    assert_eq!(source_label(r), "atom references");
    for other in [Source::Surrogate, Source::Fallback, Source::Dft] {
        assert_ne!(source_colour(r), source_colour(other), "{other:?}");
        assert_ne!(source_label(r), source_label(other));
        assert_eq!(source_tip(other), SOURCE_TIP);
    }
    let tip = source_tip(r);
    for want in ["lone atoms: answered from the atom references", "grid and box artefacts", "measured for a lone H", "egg-box ripple", "Dirichlet boundary", "not the container walls", "2 mHa 4 Bohr from the nearest face", "Nothing is learned", "none is audited", "draft mode answers lone atoms with the model, the same number"] {
        assert!(tip.contains(want), "the tip lacks '{want}': {tip}");
    }
    assert!(SOURCE_TIP.contains("atom references (teal)"));
}

/// File ▸ Import data from a folder… (installer Task 1): outside the installed build (these tests
/// run in dev mode) the item is there but disabled, a click sends nothing, and the action itself
/// is refused with the tooltip's reason; a finished import's line reaches the status line, and
/// a thread that died says so. The search finds it.
#[test]
fn import_data_is_offered_only_in_the_installed_build() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let snap = snapshot(&scene, Backend::Gpu, None);
    let inp = input(&scene, Some(&snap), &cam);
    let ctx = egui::Context::default();
    apply_theme(&ctx);
    let mut st = UiState::new("h2", &scene);
    assert_eq!(crate::paths::mode(), crate::paths::DataMode::Dev);
    egui::Popup::open_id(&ctx, menus::popup_id(menus::MenuId::File));
    let outs = click_widget(&ctx, &mut st, &inp, "import-data");
    egui::Popup::close_all(&ctx);
    assert!(outs.iter().all(|o| o.actions.is_empty() && o.commands.is_empty()), "a disabled item sends nothing");
    assert!(st.import_data.is_none());
    let mut out = PanelOutput::default();
    apply(&mut st, &inp, Action::ImportData("C:/old copy".into()), &mut out);
    assert_eq!(out.error.as_deref(), Some("import data: only the installed app keeps its data in %LOCALAPPDATA%"));
    assert!(out.commands.is_empty() && st.import_data.is_none());
    // the thread's line, when it arrives
    let (tx, rx) = std::sync::mpsc::channel();
    st.import_data = Some(rx);
    poll_import(&mut st);
    assert!(st.import_data.is_some(), "still running");
    tx.send("imported models from C:/old copy: 3 files copied, 0 already there (kept)".to_string()).unwrap();
    poll_import(&mut st);
    assert!(st.import_data.is_none());
    assert_eq!(st.status, "imported models from C:/old copy: 3 files copied, 0 already there (kept)");
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    st.import_data = Some(rx);
    drop(tx);
    poll_import(&mut st);
    assert!(st.import_data.is_none() && st.status.contains("stopped before it finished"), "{}", st.status);
    let r = search::results("import data", 3);
    assert!(r.first().is_some_and(|e| e.anchor == "m-files"), "{:?}", r.iter().map(|e| e.name).collect::<Vec<_>>());
    // the item's state, by mode (fix round 1, M5)
    use crate::paths::{DataMode, IMPORT_ONLY_INSTALLED};
    assert_eq!(menus::import_item(DataMode::Dev, false), Err(IMPORT_ONLY_INSTALLED));
    assert_eq!(menus::import_item(DataMode::Portable, false), Err(IMPORT_ONLY_INSTALLED));
    assert_eq!(menus::import_item(DataMode::Installed, false), Ok(()));
    assert_eq!(menus::import_item(DataMode::Installed, true), Err("an import is running"));
}

/// Fix round 1 (M5): the Installed branch of File ▸ Import data from a folder…, with the mode and a
/// temporary root standing in on `UiState`: the import starts on its thread, the status line says
/// so, a second one is refused while it runs, its line arrives with the counts, the File window
/// reads its folders again, and an import of the root itself is refused.
#[test]
fn import_data_runs_in_the_installed_build() {
    let scene = Scene::preset("h2").unwrap();
    let cam = camera();
    let inp = input(&scene, None, &cam);
    let tmp = std::env::temp_dir().join(format!("atomic-import-ui-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let (old, root) = (tmp.join("old copy"), tmp.join("Local").join("atomic-sandbox"));
    std::fs::create_dir_all(old.join("scenes")).unwrap();
    std::fs::write(old.join("scenes").join("w.scene"), "grid 33 0.25\n").unwrap();
    let mut st = UiState::new("h2", &scene);
    (st.data_mode, st.data_root) = (crate::paths::DataMode::Installed, Some(root.clone()));
    assert!(!st.files.refresh_pending());
    let mut out = PanelOutput::default();
    apply(&mut st, &inp, Action::ImportData(old.clone()), &mut out);
    assert_eq!(out.error, None);
    assert!(st.import_data.is_some() && st.status.starts_with("importing models, profiles and scenes from"), "{}", st.status);
    let mut again = PanelOutput::default();
    apply(&mut st, &inp, Action::ImportData(old.clone()), &mut again);
    assert_eq!(again.error.as_deref(), Some("import data: an import is already running"));
    let until = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while st.import_data.is_some() && std::time::Instant::now() < until {
        poll_import(&mut st);
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(st.import_data.is_none(), "the import finished");
    assert!(st.status.starts_with("imported scenes from") && st.status.contains("1 file copied"), "{}", st.status);
    assert_eq!(std::fs::read_to_string(root.join("scenes").join("w.scene")).unwrap(), "grid 33 0.25\n");
    assert!(root.join(crate::paths::MIGRATED_FROM).is_file());
    assert!(st.files.refresh_pending(), "the File window reads its folders again");
    // the root itself (or a folder inside it) is not an older copy
    let mut out = PanelOutput::default();
    apply(&mut st, &inp, Action::ImportData(root.join("scenes")), &mut out);
    assert!(out.error.as_deref().is_some_and(|e| e.contains("is inside this build's own data folder")), "{:?}", out.error);
    assert!(st.import_data.is_none());
    let _ = std::fs::remove_dir_all(&tmp);
}

// ---- the Version panel, the chip and the restart guard (installer Task 4) ----
//
// No test touches the network: `UpdateState`'s default network is `version::Offline` in a test
// build, and the tests that need answers stand in `FixtureNet` (the recorded releases, the signed
// fixture sums, a stand-in installer from memory).

mod updates {
    use super::*;
    use crate::update::{Channel, CheckError, DownloadState, Formats, Release};
    use std::io::Read;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use version::{Guard, UpdateNet, PORTABLE_REFUSED, UPDATES_OFF};

    const RELEASES: &str = include_str!("../update/fixtures/releases.json");
    const SUMS: &[u8] = include_bytes!("../update/fixtures/SHA256SUMS.txt");
    const SIG: &[u8] = include_bytes!("../update/fixtures/SHA256SUMS.txt.minisig");
    const TEST_KEY: &str = include_str!("../update/fixtures/TEST-ONLY.pub");
    /// The bytes the fixture `SHA256SUMS.txt` lists as the v0.2.0 installer (`update::apply`'s tests).
    const FIXTURE_SETUP: &[u8] = b"AtomicSandbox-Setup-0.2.0.exe: a stand-in for the installer, test only\n";

    pub(super) fn releases() -> Vec<Release> {
        crate::update::release::parse_releases(RELEASES).unwrap()
    }

    pub(super) fn ver(s: &str) -> semver::Version {
        semver::Version::parse(s).unwrap()
    }

    /// The recorded answers: the releases list (or a refusal), a manifest's formats, and the
    /// installer from memory — held back while `gate` is set (a stalled connection). `stalled` is
    /// set once a download's read is held at the gate, so a test can wait for that state instead
    /// of racing the job's thread.
    pub(super) struct FixtureNet {
        pub list: Result<Vec<Release>, CheckError>,
        pub manifest: Result<Option<Formats>, String>,
        pub gate: Arc<AtomicBool>,
        pub stalled: Arc<AtomicBool>,
        pub calls: Arc<AtomicUsize>,
    }

    impl FixtureNet {
        pub fn new() -> FixtureNet {
            FixtureNet { list: Ok(releases()), manifest: Ok(None), gate: Arc::new(AtomicBool::new(false)), stalled: Arc::new(AtomicBool::new(false)), calls: Arc::new(AtomicUsize::new(0)) }
        }
    }

    /// The installer's bytes, after the gate opens (`stalled` set while a read waits at it).
    struct Gated {
        gate: Arc<AtomicBool>,
        stalled: Arc<AtomicBool>,
        bytes: &'static [u8],
    }

    impl Read for Gated {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            while self.gate.load(Ordering::Relaxed) {
                self.stalled.store(true, Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            let n = self.bytes.len().min(buf.len()).min(16);
            buf[..n].copy_from_slice(&self.bytes[..n]);
            self.bytes = &self.bytes[n..];
            Ok(n)
        }
    }

    impl UpdateNet for FixtureNet {
        fn releases(&self) -> Result<Vec<Release>, CheckError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.list.clone()
        }
        fn manifest(&self, _: &Release) -> Result<Option<Formats>, String> {
            self.manifest.clone()
        }
        fn download(&self, r: &Release, dir: std::path::PathBuf) -> Result<crate::update::Download, String> {
            let (gate, stalled) = (self.gate.clone(), self.stalled.clone());
            let setup = crate::update::release::installer_name(&r.version);
            let source = move |url: &str| -> Result<Box<dyn Read + Send>, String> {
                match url.rsplit('/').next().unwrap() {
                    "SHA256SUMS.txt" => Ok(Box::new(SUMS)),
                    "SHA256SUMS.txt.minisig" => Ok(Box::new(SIG)),
                    n if n == setup => Ok(Box::new(Gated { gate: gate.clone(), stalled: stalled.clone(), bytes: FIXTURE_SETUP })),
                    n => Err(format!("no {n}")),
                }
            };
            // the stand-in's size, as listed
            let mut r = r.clone();
            let name = crate::update::release::installer_name(&r.version);
            r.assets.iter_mut().filter(|a| a.name == name).for_each(|a| a.size = FIXTURE_SETUP.len() as u64);
            crate::update::Download::start_with(&r, dir, Box::new(source), TEST_KEY)
        }
    }

    /// A temporary per-user root for an installed build standing in (its `updates\` holds the
    /// downloads).
    pub(super) fn temp_root(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("atomic-version-ui-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// `st` as an installed build with `root`, the fixture list, this build at `current`, and the
    /// fixture network.
    pub(super) fn installed(st: &mut UiState, root: &std::path::Path, current: &str) -> Arc<AtomicBool> {
        installed_gated(st, root, current).0
    }

    /// [`installed`], with the fixture network's `gate` and `stalled` flags ([`FixtureNet`]).
    fn installed_gated(st: &mut UiState, root: &std::path::Path, current: &str) -> (Arc<AtomicBool>, Arc<AtomicBool>) {
        let net = FixtureNet::new();
        let flags = (net.gate.clone(), net.stalled.clone());
        (st.data_mode, st.data_root) = (crate::paths::DataMode::Installed, Some(root.to_path_buf()));
        st.update.net = Arc::new(net);
        st.update.releases = releases();
        st.update.current = ver(current);
        flags
    }

    /// Polls `done` every 2 ms until it holds, for at most `secs` (a generous bound: a loaded
    /// machine only makes it slower, never changes the outcome); whether it held.
    fn wait_for(secs: u64, mut done: impl FnMut() -> bool) -> bool {
        let t = std::time::Instant::now();
        while !done() {
            if t.elapsed().as_secs() >= secs {
                return done();
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        true
    }

    /// Runs frames until the download is no longer running (at most 60 s).
    fn wait_download(st: &mut UiState) -> DownloadState {
        let t = std::time::Instant::now();
        loop {
            let s = st.update.download.as_ref().expect("a download").poll();
            if !matches!(s, DownloadState::Running(..)) || t.elapsed().as_secs() > 60 {
                return s;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    /// Every text drawn in the third of three frames (a window sizes itself first).
    pub(super) fn shown(ctx: &egui::Context, st: &mut UiState, inp: &PanelInput<'_>) -> Vec<String> {
        fn walk(s: &egui::epaint::Shape, out: &mut Vec<String>) {
            match s {
                egui::epaint::Shape::Text(t) => out.push(t.galley.text().to_string()),
                egui::epaint::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                _ => {}
            }
        }
        let mut out = Vec::new();
        for _ in 0..3 {
            let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0))), ..Default::default() };
            let mut full = ctx.run_ui(raw, |root| {
                show(root, st, inp);
            });
            full.textures_delta.clear();
            out.clear();
            full.shapes.iter().for_each(|c| walk(&c.shape, &mut out));
        }
        out
    }

    fn has(lines: &[String], text: &str) -> bool {
        lines.iter().any(|l| l.contains(text))
    }

    fn line_at(lines: &[String], text: &str) -> usize {
        lines.iter().position(|l| l == text).unwrap_or_else(|| panic!("no line {text:?} in {lines:#?}"))
    }

    fn actions(outs: &[PanelOutput]) -> Vec<Action> {
        outs.iter().flat_map(|o| o.actions.clone()).collect()
    }

    fn probed_now(ctx: &egui::Context, st: &mut UiState, inp: &PanelInput<'_>, name: &str) -> bool {
        PROBES.with(|p| p.borrow_mut().clear());
        shown(ctx, st, inp);
        PROBES.with(|p| p.borrow().contains_key(name))
    }

    fn ctx() -> egui::Context {
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        ctx
    }

    /// The chip offers the newest release on the channel that is newer than this build — none
    /// without a list, none when this build is the newest, a newer pre-release on Beta only — and a
    /// click on it opens the Version panel.
    #[test]
    fn chip_shows_only_when_newer_on_channel() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = ctx();
        let mut st = UiState::new("h2", &scene);
        assert_eq!(version::chip_text(&st.update), None, "no list yet");
        assert!(!probed_now(&ctx, &mut st, &inp, "update-chip"));
        st.update.releases = releases();
        st.update.current = ver("0.1.0");
        assert_eq!(version::chip_text(&st.update).as_deref(), Some("v0.2.0 available"));
        assert!(has(&shown(&ctx, &mut st, &inp), "v0.2.0 available"), "drawn in the top bar");
        click_widget(&ctx, &mut st, &inp, "update-chip");
        assert!(st.update.open, "the chip opens the Version panel");
        st.update.open = false;
        st.update.current = ver("0.2.0");
        for ch in [Channel::Stable, Channel::Beta] {
            st.update.channel = ch;
            assert_eq!(version::chip_text(&st.update), None, "{ch:?}: v0.2.0 is the newest (its beta is older)");
        }
        assert!(!probed_now(&ctx, &mut st, &inp, "update-chip"));
        // a pre-release newer than every release: Beta's chip only
        let mut beta = releases().into_iter().find(|r| r.tag == "v0.2.0-beta.1").unwrap();
        (beta.tag, beta.version) = ("v0.3.0-beta.1".into(), ver("0.3.0-beta.1"));
        st.update.releases.insert(0, beta);
        st.update.channel = Channel::Stable;
        assert_eq!(version::chip_text(&st.update), None);
        assert!(!probed_now(&ctx, &mut st, &inp, "update-chip"));
        st.update.channel = Channel::Beta;
        assert_eq!(version::chip_text(&st.update).as_deref(), Some("v0.3.0-beta.1 available"));
        assert!(probed_now(&ctx, &mut st, &inp, "update-chip"));
    }

    /// Help ▸ Version… opens the panel: every release of the channel newest first (the draft not
    /// listed), with its date, the running one marked "installed", the notes collapsed.
    #[test]
    fn version_panel_lists_newest_first_and_marks_installed() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = ctx();
        let mut st = UiState::new("h2", &scene);
        st.update.releases = releases();
        st.update.current = ver("0.1.0");
        st.update.channel = Channel::Beta;
        egui::Popup::open_id(&ctx, menus::popup_id(menus::MenuId::Help));
        click_widget(&ctx, &mut st, &inp, "help-version");
        egui::Popup::close_all(&ctx);
        assert!(st.update.open, "Help ▸ Version… opens it");
        let tags: Vec<&str> = st.update.listed().iter().map(|r| r.tag.as_str()).collect();
        assert_eq!(tags, ["v0.2.0", "v0.2.0-beta.1", "v0.1.0"], "newest first; the draft v0.3.0 is not public");
        let lines = shown(&ctx, &mut st, &inp);
        let (a, b, c) = (line_at(&lines, "v0.2.0"), line_at(&lines, "v0.2.0-beta.1"), line_at(&lines, "v0.1.0"));
        assert!(a < b && b < c, "on screen newest first: {a} {b} {c}");
        assert_eq!(lines.iter().filter(|l| *l == "installed").count(), 1, "one entry is the running one");
        let installed = line_at(&lines, "installed");
        assert!(installed > c, "v0.1.0 is marked installed");
        for date in ["2026-12-01", "2026-11-20", "2026-11-01"] {
            assert!(lines.iter().any(|l| l == date), "{date}");
        }
        assert!(has(&lines, "release notes"), "the notes, collapsed");
        assert!(!has(&lines, "Installer and in-app updates"), "collapsed: the body is not drawn");
        assert!(has(&lines, "Atomic Sandbox v0.1.0"));
        // the window's close button closes it
        st.update.open = false;
        assert!(!has(&shown(&ctx, &mut st, &inp), "v0.2.0-beta.1"));
    }

    /// Stable lists releases only; Beta (its button) lists the pre-release too, marked "beta".
    #[test]
    fn beta_channel_shows_prerelease() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = ctx();
        let mut st = UiState::new("h2", &scene);
        // the fixture's oldest release, not this build's version (the panel's title names it)
        st.update.current = ver("0.1.0");
        st.update.releases = releases();
        st.update.open = true;
        assert_eq!(st.update.channel, Channel::Stable, "Stable by default");
        assert!(st.update.listed().iter().all(|r| !r.prerelease));
        let lines = shown(&ctx, &mut st, &inp);
        assert!(!has(&lines, "v0.2.0-beta.1") && !lines.iter().any(|l| l == "beta"), "{lines:#?}");
        click_widget(&ctx, &mut st, &inp, "channel-beta");
        assert_eq!(st.update.channel, Channel::Beta);
        let lines = shown(&ctx, &mut st, &inp);
        assert!(lines.iter().any(|l| l == "v0.2.0-beta.1") && lines.iter().any(|l| l == "beta"), "{lines:#?}");
        click_widget(&ctx, &mut st, &inp, "channel-stable");
        assert_eq!(st.update.channel, Channel::Stable);
    }

    /// An unsigned release (v0.1.0: a zip, no SHA256SUMS.txt) is listed with its Install button
    /// disabled and the reason beside it; a click sends nothing, and the action itself refuses.
    #[test]
    fn unsigned_release_button_disabled_with_reason() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = ctx();
        let mut st = UiState::new("h2", &scene);
        let root = temp_root("unsigned");
        installed(&mut st, &root, "0.2.0");
        st.update.open = true;
        let lines = shown(&ctx, &mut st, &inp);
        assert!(has(&lines, crate::update::release::NO_SIGNATURE), "{lines:#?}");
        let outs = click_widget(&ctx, &mut st, &inp, "install-v0.1.0");
        assert!(actions(&outs).is_empty(), "disabled: nothing sent");
        assert!(st.update.confirm.is_none() && st.update.download.is_none());
        let out = act(&mut st, &inp, Action::InstallVersion("v0.1.0".into()));
        assert!(out.error.as_deref().is_some_and(|e| e.contains(crate::update::release::NO_SIGNATURE)), "{:?}", out.error);
        assert!(st.update.download.is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Install and restart on an older release shows its downgrade warnings first — from its
    /// manifest when it has one (read on a thread), the generic line when it has none — and only
    /// "Install … anyway" sends `InstallVersion`, which starts the download.
    #[test]
    fn install_older_shows_downgrade_warning_first() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = ctx();
        let mut st = UiState::new("h2", &scene);
        let root = temp_root("downgrade");
        installed(&mut st, &root, "0.3.0");
        // v0.2.0 carries a manifest: one whose scene format is older than this build's
        let mut formats = crate::update::current_formats();
        formats.insert(crate::update::formats::SCENE.into(), 0);
        st.update.net = Arc::new(FixtureNet { manifest: Ok(Some(formats)), ..FixtureNet::new() });
        st.update.open = true;
        let outs = click_widget(&ctx, &mut st, &inp, "install-v0.2.0");
        assert!(actions(&outs).is_empty(), "an older target asks first: {:?}", actions(&outs));
        assert!(st.update.download.is_none(), "nothing started");
        assert_eq!(st.update.confirm.as_ref().map(|c| c.tag.as_str()), Some("v0.2.0"));
        let t = std::time::Instant::now();
        while st.update.confirm.as_ref().is_some_and(|c| c.warnings.is_none()) && t.elapsed().as_secs() < 10 {
            shown(&ctx, &mut st, &inp);
        }
        let w = st.update.confirm.as_ref().unwrap().warnings.clone().expect("the manifest was read");
        assert_eq!(w, [format!("scenes saved by this version (v{}) may not open in v0.2.0", env!("CARGO_PKG_VERSION"))]);
        let lines = shown(&ctx, &mut st, &inp);
        assert!(has(&lines, "Before going back to the older v0.2.0") && has(&lines, "scenes saved by this version"), "{lines:#?}");
        assert!(has(&lines, "Install v0.2.0 anyway"));
        let outs = click_widget(&ctx, &mut st, &inp, "install-anyway");
        assert_eq!(actions(&outs), [Action::InstallVersion("v0.2.0".into())]);
        assert!(st.update.confirm.is_none() && st.update.download.is_some(), "the download started");
        assert_eq!(wait_download(&mut st), DownloadState::Ready(root.join("updates").join("AtomicSandbox-Setup-0.2.0.exe")));
        // no manifest: the generic warning, at once
        st.update.download = None;
        let mut no_manifest = releases();
        no_manifest.iter_mut().for_each(|r| r.assets.retain(|a| a.name != "manifest.json"));
        st.update.releases = no_manifest;
        click_widget(&ctx, &mut st, &inp, "install-v0.2.0");
        let w = st.update.confirm.as_ref().unwrap().warnings.clone().expect("no manifest to read");
        assert_eq!(w, ["v0.2.0 records no data formats: collections, models or profiles saved by this version may not open in it"]);
        assert!(has(&shown(&ctx, &mut st, &inp), "v0.2.0 records no data formats"));
        click_widget(&ctx, &mut st, &inp, "install-cancel");
        assert!(st.update.confirm.is_none() && st.update.download.is_none(), "Cancel: nothing starts");
        // a newer target asks nothing
        st.update.current = ver("0.1.0");
        let outs = click_widget(&ctx, &mut st, &inp, "install-v0.2.0");
        assert_eq!(actions(&outs), [Action::InstallVersion("v0.2.0".into())]);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A ready download in an installed build: `st` with v0.2.0 downloaded and verified.
    fn ready(st: &mut UiState, inp: &PanelInput<'_>, root: &std::path::Path) {
        installed(st, root, "0.1.0");
        assert!(act(st, inp, Action::InstallVersion("v0.2.0".into())).error.is_none());
        assert!(matches!(wait_download(st), DownloadState::Ready(_)));
        assert!(st.update.verified().is_some());
    }

    /// Restart now while a refit runs waits ("Waiting for the refit to finish…", Cancel it and
    /// restart / Keep waiting) and restarts by itself once it ends; Cancel it and restart goes at
    /// once; a sampling pass and a background audit are waited for too (a paused solve is not a
    /// background job, but is waited for: `restart_waits_for_a_paused_solve`); nothing restarts
    /// without a verified download.
    #[test]
    fn restart_waits_for_running_refit() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let idle = input(&scene, None, &cam);
        let refit = PanelInput { probe: Some(sim::ProbeView { refit: Some((3.0, 120)), ..Default::default() }), ..input(&scene, None, &cam) };
        let ctx = ctx();
        let mut st = UiState::new("h2", &scene);
        let root = temp_root("restart");
        installed(&mut st, &root, "0.1.0");
        let out = act(&mut st, &idle, Action::RestartToUpdate);
        assert!(!out.restart_for_update && out.error.as_deref().is_some_and(|e| e.contains("nothing is downloaded")), "{:?}", out.error);
        ready(&mut st, &idle, &root);
        st.update.open = true;
        let out = act(&mut st, &refit, Action::RestartToUpdate);
        assert!(!out.restart_for_update);
        assert_eq!(st.update.guard, Guard::Waiting { what: "the refit", prompt: true });
        let lines = shown(&ctx, &mut st, &refit);
        for t in ["Waiting for the refit to finish…", "Cancel it and restart", "Keep waiting", "goes ahead when it ends"] {
            assert!(has(&lines, t), "{t:?} in {lines:#?}");
        }
        assert!(!has(&lines, "nothing is cut off") && !has(&lines, "installer will ask"), "{lines:#?}");
        click_widget(&ctx, &mut st, &refit, "guard-keep-waiting");
        assert_eq!(st.update.guard, Guard::Waiting { what: "the refit", prompt: false });
        let lines = shown(&ctx, &mut st, &refit);
        assert!(has(&lines, "Restarting when the refit finishes…") && has(&lines, "Closing this window calls the restart off"));
        assert!(!frame(&ctx, &mut st, &refit, 1400.0, vec![]).restart_for_update, "still running: still waiting");
        // the refit ends: the restart goes ahead by itself
        assert!(frame(&ctx, &mut st, &idle, 1400.0, vec![]).restart_for_update);
        assert_eq!(st.update.guard, Guard::Idle);
        // Cancel it and restart: at once, the refit still running
        act(&mut st, &refit, Action::RestartToUpdate);
        let outs = click_widget(&ctx, &mut st, &refit, "guard-cancel-restart");
        assert_eq!(actions(&outs), [Action::RestartToUpdate]);
        assert!(outs.iter().any(|o| o.restart_for_update));
        assert_eq!(st.update.guard, Guard::Idle);
        // a sampling pass and a background audit are waited for (the pass's model holds no sample
        // yet here, so nothing is asked about it first)
        let mut m = model();
        m.samples = 0;
        let mut snap = snapshot(&scene, Backend::Estimated, Some(m));
        snap.dynamics.sampling = Some(sim::SamplingProgress { stage: "perturbations", done: 3, total: 40, skipped: 0, failed: 0, cancelled: false, last_energy: f64::NAN, finished: false, target: None });
        let sampling = input(&scene, Some(&snap), &cam);
        assert_eq!(version::running(&sampling), Some("the sampling pass"));
        act(&mut st, &sampling, Action::RestartToUpdate);
        assert_eq!(st.update.guard, Guard::Waiting { what: "the sampling pass", prompt: true });
        let audit = PanelInput { probe: Some(sim::ProbeView { side: sim::activity::SlotView { running: true, ..Default::default() }, ..Default::default() }), ..input(&scene, None, &cam) };
        assert_eq!(version::running(&audit), Some("the background audit"));
        // a paused step solve in flight is not a background job, though the restart waits for it
        let solving = PanelInput { busy: true, probe: Some(sim::ProbeView { main: sim::activity::SlotView { running: true, ..Default::default() }, ..Default::default() }), ..input(&scene, None, &cam) };
        assert_eq!(version::running(&solving), None);
        assert_eq!(version::waiting_for(&solving), Some("the solve"));
        // Cancel (the download) calls a waiting restart off
        act(&mut st, &idle, Action::CancelUpdateDownload);
        assert_eq!(st.update.guard, Guard::Idle);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// An unsaved in-session model with no collection: Restart now first offers Save model… /
    /// Restart anyway / Cancel, saying what would be lost; Restart anyway goes on; Save model…
    /// opens the Save window; a model in a shared collection is kept, so nothing is asked.
    #[test]
    fn unsaved_model_offers_save_first() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let snap = snapshot(&scene, Backend::Estimated, Some(model()));
        let inp = input(&scene, Some(&snap), &cam);
        let ctx = ctx();
        let mut st = UiState::new("h2", &scene);
        assert!(version::save_note(&st.update, &inp).is_some_and(|t| t.contains("(273 samples) isn't in a shared collection or a saved file")), "273 samples, no collection, never saved");
        st.backend = Backend::Estimated;
        let root = temp_root("unsaved");
        ready(&mut st, &inp, &root);
        st.update.open = true;
        let out = act(&mut st, &inp, Action::RestartToUpdate);
        assert!(!out.restart_for_update);
        assert_eq!(st.update.guard, Guard::SaveFirst);
        let lines = shown(&ctx, &mut st, &inp);
        for t in ["The restart won't keep what this model learned", "(273 samples)", "Save model…", "Restart anyway", "Cancel"] {
            assert!(has(&lines, t), "{t:?} in {lines:#?}");
        }
        click_widget(&ctx, &mut st, &inp, "guard-cancel");
        assert_eq!(st.update.guard, Guard::Idle);
        act(&mut st, &inp, Action::RestartToUpdate);
        click_widget(&ctx, &mut st, &inp, "guard-save");
        assert!(st.files.save_open, "the Save window, for the model");
        assert_eq!(st.update.guard, Guard::Idle);
        st.files.save_open = false;
        act(&mut st, &inp, Action::RestartToUpdate);
        let outs = click_widget(&ctx, &mut st, &inp, "guard-anyway");
        assert_eq!(actions(&outs), [Action::RestartToUpdate]);
        assert!(outs.iter().any(|o| o.restart_for_update), "Restart anyway goes on");
        assert!(!st.update.model_ack && !st.update.force, "the flags are spent");
        // a shared collection keeps the model: no question
        let mut m = model();
        m.source = sim::ModelSource::Collection { slug: "h2".into(), samples: 273, read_only: false };
        let snap = snapshot(&scene, Backend::Estimated, Some(m));
        let kept = input(&scene, Some(&snap), &cam);
        assert_eq!(version::save_note(&st.update, &kept), None);
        assert!(act(&mut st, &kept, Action::RestartToUpdate).restart_for_update);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Ruling M1: Restart now asks only about samples learned since the model was last loaded from
    /// or saved to a file — a model just loaded, or just saved, is not asked about; ten more samples
    /// are, by count; the session's autosave is not the user's file.
    #[test]
    fn save_first_asks_only_about_what_was_learned_since_load_or_save() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let ctx = ctx();
        let mut st = UiState::new("h2", &scene);
        let with = |samples: usize, source: sim::ModelSource, autosave: bool| {
            let mut m = model();
            (m.samples, m.source, m.source_is_autosave) = (samples, source, autosave);
            snapshot(&scene, Backend::Estimated, Some(m))
        };
        let file = sim::ModelSource::File("C:/models/h2 water.model".into());
        // loaded from a file: the frame sees the new source and takes its 273 as the baseline
        let loaded = with(273, file.clone(), false);
        let inp = input(&scene, Some(&loaded), &cam);
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        assert_eq!(st.update.model_baseline, Some(273));
        assert_eq!(version::save_note(&st.update, &inp), None, "just loaded: nothing to lose");
        let more = with(283, file.clone(), false);
        let inp = input(&scene, Some(&more), &cam);
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        let note = version::save_note(&st.update, &inp).expect("10 learned since the load");
        assert!(note.contains("The model from h2 water.model has learned 10 samples since it was loaded or saved") && note.contains("the file itself stays as it was"), "{note}");
        // Save model: what it holds now is in a file
        act(&mut st, &inp, Action::SaveModel);
        assert_eq!((st.update.model_baseline, version::save_note(&st.update, &inp)), (Some(283), None));
        // a session's model saved once, then learning on
        let mut st = UiState::new("h2", &scene);
        let session = with(100, sim::ModelSource::None, false);
        let inp = input(&scene, Some(&session), &cam);
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        act(&mut st, &inp, Action::SaveModel);
        let grown = with(101, sim::ModelSource::None, false);
        let inp = input(&scene, Some(&grown), &cam);
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        assert!(version::save_note(&st.update, &inp).is_some_and(|t| t.contains("has learned 1 sample since you last saved it")));
        // forgotten and learned anew (fewer samples than the baseline): another model, all of it
        let anew = with(40, sim::ModelSource::None, false);
        let inp = input(&scene, Some(&anew), &cam);
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        assert!(version::save_note(&st.update, &inp).is_some_and(|t| t.contains("(40 samples) isn't in a shared collection")));
        // the solver switch's autosave is removed at the exit: not a file of the user's
        let mut st = UiState::new("h2", &scene);
        let auto = with(50, sim::ModelSource::File("C:/temp/autosave.model".into()), true);
        let inp = input(&scene, Some(&auto), &cam);
        frame(&ctx, &mut st, &inp, 1400.0, vec![]);
        assert_eq!(st.update.model_baseline, None);
        assert!(version::save_note(&st.update, &inp).is_some_and(|t| t.contains("(50 samples)")));
    }

    /// Ruling I2: Restart now never pauses by itself. A playing run gets "The simulation is
    /// playing." with Pause and restart / Cancel; Pause and restart sends the pause (the recorded
    /// Pause action) and the restart follows once the run has stopped; a background job after the
    /// pause is still waited for.
    #[test]
    fn a_playing_run_is_paused_only_by_its_own_click() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let playing = PanelInput { playing: true, busy: true, ..input(&scene, None, &cam) };
        let stopping = PanelInput { playing: false, busy: true, ..input(&scene, None, &cam) };
        let stopped = input(&scene, None, &cam);
        let ctx = ctx();
        let mut st = UiState::new("h2", &scene);
        let root = temp_root("playing");
        ready(&mut st, &stopped, &root);
        st.update.open = true;
        let out = act(&mut st, &playing, Action::RestartToUpdate);
        assert!(out.commands.is_empty() && !out.restart_for_update, "nothing sent by itself: {:?}", out.commands);
        assert_eq!(st.update.guard, Guard::Playing);
        let lines = shown(&ctx, &mut st, &playing);
        for t in [version::PLAYING, "Pause and restart", "Cancel", "the step being solved is dropped"] {
            assert!(has(&lines, t), "{t:?} in {lines:#?}");
        }
        click_widget(&ctx, &mut st, &playing, "guard-cancel");
        assert_eq!(st.update.guard, Guard::Idle);
        act(&mut st, &playing, Action::RestartToUpdate);
        let outs = click_widget(&ctx, &mut st, &playing, "guard-pause-restart");
        assert_eq!(actions(&outs), [Action::Pause], "the pause, as the Pause button sends it");
        assert!(dyn_commands(&outs).iter().any(|c| matches!(c, DynCommand::Pause)));
        assert!(!outs.iter().any(|o| o.restart_for_update), "not before the run has stopped");
        assert_eq!(st.update.guard, Guard::Pausing { landed: false }, "the pause is on its way (the click's frames still read playing)");
        assert!(has(&shown(&ctx, &mut st, &playing), "Pausing, then restarting…"));
        assert!(!frame(&ctx, &mut st, &stopping, 1400.0, vec![]).restart_for_update, "the step still lets go of the solver");
        assert!(frame(&ctx, &mut st, &stopped, 1400.0, vec![]).restart_for_update, "stopped: the restart follows");
        assert_eq!(st.update.guard, Guard::Idle);
        // paused onto a running refit: that is still waited for
        let refit = PanelInput { probe: Some(sim::ProbeView { refit: Some((1.0, 50)), ..Default::default() }), ..input(&scene, None, &cam) };
        act(&mut st, &playing, Action::RestartToUpdate);
        click_widget(&ctx, &mut st, &playing, "guard-pause-restart");
        assert!(!frame(&ctx, &mut st, &refit, 1400.0, vec![]).restart_for_update);
        assert_eq!(st.update.guard, Guard::Waiting { what: "the refit", prompt: true });
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Final review I1: Restart now with the run paused but the worker still solving (the solve
    /// after a scene load, an Apply grid or a Step; Bonds now)
    /// waits for "the solve" with the same card, hands nothing off meanwhile, and restarts by
    /// itself once the worker is idle; Cancel it and restart goes at once; Play pressed while it
    /// waits turns it into the playing question.
    #[test]
    fn restart_waits_for_a_paused_solve() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let solving = PanelInput { busy: true, ..input(&scene, None, &cam) };
        let playing = PanelInput { playing: true, busy: true, ..input(&scene, None, &cam) };
        let idle = input(&scene, None, &cam);
        let ctx = ctx();
        let mut st = UiState::new("h2", &scene);
        let root = temp_root("paused-solve");
        ready(&mut st, &idle, &root);
        st.update.open = true;
        let out = act(&mut st, &solving, Action::RestartToUpdate);
        assert!(!out.restart_for_update, "no hand-off while the solve runs");
        assert_eq!(st.update.guard, Guard::Waiting { what: "the solve", prompt: true });
        let lines = shown(&ctx, &mut st, &solving);
        for t in ["Waiting for the solve to finish…", "Cancel it and restart", "Keep waiting"] {
            assert!(has(&lines, t), "{t:?} in {lines:#?}");
        }
        for _ in 0..3 {
            assert!(!frame(&ctx, &mut st, &solving, 1400.0, vec![]).restart_for_update, "still solving: still waiting");
        }
        assert_eq!(st.update.guard, Guard::Waiting { what: "the solve", prompt: true });
        // the worker is idle: the restart goes ahead by itself
        assert!(frame(&ctx, &mut st, &idle, 1400.0, vec![]).restart_for_update);
        assert_eq!(st.update.guard, Guard::Idle);
        // Cancel it and restart: at once, the solve still running
        act(&mut st, &solving, Action::RestartToUpdate);
        let outs = click_widget(&ctx, &mut st, &solving, "guard-cancel-restart");
        assert!(outs.iter().any(|o| o.restart_for_update));
        assert_eq!(st.update.guard, Guard::Idle);
        // Play pressed while waiting: the playing question, not a restart once the run stops
        act(&mut st, &solving, Action::RestartToUpdate);
        assert!(!frame(&ctx, &mut st, &playing, 1400.0, vec![]).restart_for_update);
        assert_eq!(st.update.guard, Guard::Playing);
        assert!(!frame(&ctx, &mut st, &idle, 1400.0, vec![]).restart_for_update, "paused again: no restart without a click");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Final re-review R1: Bonds now and the estimated solver's graph
    /// solve run with the worker not `busy`; the probe's Main slot shows the solve, and the restart
    /// waits for it the same way, then goes ahead once the slot is idle.
    #[test]
    fn restart_waits_for_a_bond_solve_the_worker_does_not_count() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let main = sim::activity::SlotView { running: true, ..Default::default() };
        let bonds = PanelInput { probe: Some(sim::ProbeView { main, ..Default::default() }), ..input(&scene, None, &cam) };
        assert!(!bonds.busy);
        let idle = input(&scene, None, &cam);
        let ctx = ctx();
        let mut st = UiState::new("h2", &scene);
        let root = temp_root("bond-solve");
        ready(&mut st, &idle, &root);
        st.update.open = true;
        assert!(!act(&mut st, &bonds, Action::RestartToUpdate).restart_for_update, "no hand-off during the bond solve");
        assert_eq!(st.update.guard, Guard::Waiting { what: "the solve", prompt: true });
        assert!(!frame(&ctx, &mut st, &bonds, 1400.0, vec![]).restart_for_update);
        assert!(frame(&ctx, &mut st, &idle, 1400.0, vec![]).restart_for_update, "the slot idle: the restart goes ahead");
        assert_eq!(st.update.guard, Guard::Idle);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// T4 N5 and N3: Play pressed while Pause and restart is pausing asks again (the playing
    /// question), so a later plain Pause never restarts the app; and a "Cancel it and restart"
    /// given for a job is spent when the guard turns to the playing question, so the job is
    /// waited for again after the next pause.
    #[test]
    fn play_while_pausing_asks_again() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let playing = PanelInput { playing: true, busy: true, ..input(&scene, None, &cam) };
        let stopping = PanelInput { playing: false, busy: true, ..input(&scene, None, &cam) };
        let stopped = input(&scene, None, &cam);
        let ctx = ctx();
        let mut st = UiState::new("h2", &scene);
        let root = temp_root("play-while-pausing");
        ready(&mut st, &stopped, &root);
        st.update.open = true;
        act(&mut st, &playing, Action::RestartToUpdate);
        click_widget(&ctx, &mut st, &playing, "guard-pause-restart");
        assert_eq!(st.update.guard, Guard::Pausing { landed: false });
        // the pause lands; the dropped step still holds the solver
        assert!(!frame(&ctx, &mut st, &stopping, 1400.0, vec![]).restart_for_update);
        assert_eq!(st.update.guard, Guard::Pausing { landed: true });
        // the user presses Play: asked again
        assert!(!frame(&ctx, &mut st, &playing, 1400.0, vec![]).restart_for_update);
        assert_eq!(st.update.guard, Guard::Playing);
        assert!(has(&shown(&ctx, &mut st, &playing), version::PLAYING));
        // a later plain Pause: no restart without a new click
        for inp in [&stopping, &stopped, &stopped] {
            assert!(!frame(&ctx, &mut st, inp, 1400.0, vec![]).restart_for_update, "a plain Pause never restarts");
        }
        assert_eq!(st.update.guard, Guard::Playing);
        // the new click restarts as before
        let outs = click_widget(&ctx, &mut st, &stopped, "guard-pause-restart");
        assert!(outs.iter().any(|o| o.restart_for_update), "paused already: the restart follows");
        assert_eq!(st.update.guard, Guard::Idle);
        // N3: Cancel it and restart (for the refit) while playing turns to the playing question
        // and is spent there; after Pause and restart the refit is waited for again
        let refit_playing = PanelInput { playing: true, busy: true, probe: Some(sim::ProbeView { refit: Some((1.0, 50)), ..Default::default() }), ..input(&scene, None, &cam) };
        let refit = PanelInput { probe: Some(sim::ProbeView { refit: Some((1.0, 50)), ..Default::default() }), ..input(&scene, None, &cam) };
        act(&mut st, &refit, Action::RestartToUpdate);
        assert_eq!(st.update.guard, Guard::Waiting { what: "the refit", prompt: true });
        let outs = click_widget(&ctx, &mut st, &refit_playing, "guard-cancel-restart");
        assert!(!outs.iter().any(|o| o.restart_for_update));
        assert_eq!(st.update.guard, Guard::Playing);
        assert!(!st.update.force, "the cancel was for the job, not for after the pause");
        click_widget(&ctx, &mut st, &refit_playing, "guard-pause-restart");
        assert!(!frame(&ctx, &mut st, &refit, 1400.0, vec![]).restart_for_update);
        assert_eq!(st.update.guard, Guard::Waiting { what: "the refit", prompt: true });
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Ruling I1: closing the Version window calls a pending restart off — waiting, pausing or
    /// asking — with a status line; the download stays ready, and Restart now is offered again.
    #[test]
    fn closing_the_panel_calls_a_pending_restart_off() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let refit = PanelInput { probe: Some(sim::ProbeView { refit: Some((3.0, 120)), ..Default::default() }), ..input(&scene, None, &cam) };
        let idle = input(&scene, None, &cam);
        let ctx = ctx();
        let mut st = UiState::new("h2", &scene);
        let root = temp_root("close");
        ready(&mut st, &idle, &root);
        st.update.open = true;
        act(&mut st, &refit, Action::RestartToUpdate);
        click_widget(&ctx, &mut st, &refit, "guard-keep-waiting");
        assert_eq!(st.update.guard, Guard::Waiting { what: "the refit", prompt: false });
        version::close_panel(&mut st);
        assert!(!st.update.open);
        assert_eq!(st.update.guard, Guard::Idle);
        assert_eq!(st.status, version::CALLED_OFF);
        assert!(st.update.verified().is_some(), "the download stays ready");
        assert!(!frame(&ctx, &mut st, &idle, 1400.0, vec![]).restart_for_update, "the refit ending restarts nothing");
        st.update.open = true;
        assert!(has(&shown(&ctx, &mut st, &idle), "Restart now"), "offered again");
        // closing with nothing pending says nothing
        st.set_status("x");
        version::close_panel(&mut st);
        assert_eq!(st.status, "x");
        // the window's own close button goes through it (the guard asking about the model)
        let snap = snapshot(&scene, Backend::Estimated, Some(model()));
        let unsaved = input(&scene, Some(&snap), &cam);
        st.update.open = true;
        act(&mut st, &unsaved, Action::RestartToUpdate);
        assert_eq!(st.update.guard, Guard::SaveFirst);
        version::close_panel(&mut st);
        assert_eq!((st.update.guard, st.status.as_str()), (Guard::Idle, version::CALLED_OFF));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Ruling I4, the reviewer's path: Restart now during a first pass (waiting), the pass learns,
    /// Cancel it and restart → the model question → Cancel; later, during another pass, Restart now
    /// → Restart anyway must wait for the pass again, not cut it off with the stale "cancel it".
    #[test]
    fn a_stale_cancel_it_never_skips_the_wait() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let pass = |samples: usize| {
            let mut m = model();
            m.samples = samples;
            let mut snap = snapshot(&scene, Backend::Estimated, Some(m));
            snap.dynamics.sampling = Some(sim::SamplingProgress { stage: "perturbations", done: 3, total: 40, skipped: 0, failed: 0, cancelled: false, last_energy: f64::NAN, finished: false, target: None });
            snap
        };
        let ctx = ctx();
        let mut st = UiState::new("h2", &scene);
        let root = temp_root("stale");
        ready(&mut st, &input(&scene, None, &cam), &root);
        st.update.open = true;
        let first = pass(0);
        let inp = input(&scene, Some(&first), &cam);
        act(&mut st, &inp, Action::RestartToUpdate);
        assert_eq!(st.update.guard, Guard::Waiting { what: "the sampling pass", prompt: true });
        let learned = pass(30);
        let inp = input(&scene, Some(&learned), &cam);
        let outs = click_widget(&ctx, &mut st, &inp, "guard-cancel-restart");
        assert!(!outs.iter().any(|o| o.restart_for_update));
        assert_eq!(st.update.guard, Guard::SaveFirst, "the pass learned: asked about the model first");
        click_widget(&ctx, &mut st, &inp, "guard-cancel");
        assert_eq!(st.update.guard, Guard::Idle);
        assert!(!st.update.force && !st.update.model_ack, "Cancel clears every flag");
        let second = pass(30);
        let inp = input(&scene, Some(&second), &cam);
        act(&mut st, &inp, Action::RestartToUpdate);
        assert_eq!(st.update.guard, Guard::SaveFirst);
        let outs = click_widget(&ctx, &mut st, &inp, "guard-anyway");
        assert!(!outs.iter().any(|o| o.restart_for_update), "Restart anyway agrees to losing the model, not to cutting the pass off");
        assert_eq!(st.update.guard, Guard::Waiting { what: "the sampling pass", prompt: true });
        // Save model… and install() clear the flags too
        st.update.force = true;
        st.update.guard = Guard::SaveFirst;
        click_widget(&ctx, &mut st, &inp, "guard-save");
        assert!((!st.update.force, !st.update.model_ack, st.update.guard) == (true, true, Guard::Idle));
        st.update.force = true;
        st.update.model_ack = true;
        st.update.download = None;
        act(&mut st, &inp, Action::InstallVersion("v0.2.0".into()));
        assert!(!st.update.force && !st.update.model_ack && st.update.guard == Guard::Idle);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Ruling M6 (and I3, M2): the viewer's hand-off writes the session first and hands nothing
    /// off when that write fails; with a good root the session is written, and the hand-off itself
    /// refuses here (the test process is a build from source) — no close, the error line says
    /// why, and no "handing over" status is claimed.
    #[test]
    fn a_failed_session_write_means_no_hand_off() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let root = temp_root("handoff");
        let mut v = crate::viewer::Viewer::new("h2").unwrap();
        ready(&mut v.ui, &input(&scene, None, &cam), &root);
        // a root that is a file: the session can't be written under it
        let blocked = root.join("not a folder");
        std::fs::write(&blocked, b"x").unwrap();
        v.ui.data_root = Some(blocked.clone());
        let status = v.ui.status.clone();
        v.apply(Action::RestartToUpdate);
        assert!(v.close_for_update.is_none(), "no hand-off");
        assert!(v.last_error.as_deref().is_some_and(|e| e.starts_with("restart to update: couldn't save the session")), "{:?}", v.last_error);
        assert_eq!(v.ui.status, status, "no status claims a hand-off");
        // a good root: the session is written, then the hand-off refuses (not an installed process)
        v.ui.data_root = Some(root.clone());
        v.last_error = None;
        v.apply(Action::RestartToUpdate);
        assert!(root.join("updates").join(crate::update::apply::SESSION_FILE).is_file(), "the session first");
        assert!(v.close_for_update.is_none());
        assert_eq!(v.last_error.as_deref(), Some(format!("restart to update: {UPDATES_OFF}").as_str()));
        assert_eq!(v.ui.status, status);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A portable copy offers each release's zip as a link (this repository's own asset), no
    /// install button and no downgrade warnings; the actions refuse.
    #[test]
    fn portable_mode_shows_download_links() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = ctx();
        let mut st = UiState::new("h2", &scene);
        st.data_mode = crate::paths::DataMode::Portable;
        st.update.releases = releases();
        st.update.current = ver("0.3.0");
        st.update.open = true;
        let lines = shown(&ctx, &mut st, &inp);
        assert_eq!(lines.iter().filter(|l| *l == "Download zip").count(), 2, "v0.2.0 and v0.1.0: {lines:#?}");
        assert!(!has(&lines, "Install and restart") && !has(&lines, crate::update::release::NO_SIGNATURE));
        assert!(PROBES.with(|p| p.borrow().contains_key("zip-v0.2.0") && p.borrow().contains_key("zip-v0.1.0")));
        let r = st.update.releases.iter().find(|r| r.tag == "v0.2.0").unwrap();
        assert_eq!(version::zip_link(r), Some("https://github.com/ItsKodas/atomic-sandbox/releases/download/v0.2.0/AtomicSandbox-0.2.0-windows-x64.zip"));
        // a zip whose address isn't its release's own is no link
        let mut moved = r.clone();
        moved.assets.iter_mut().filter(|a| a.name.ends_with(".zip")).for_each(|a| a.url = "https://example.com/a.zip".into());
        assert_eq!(version::zip_link(&moved), None);
        click_widget(&ctx, &mut st, &inp, "zip-v0.2.0");
        assert!(st.update.confirm.is_none() && st.update.download.is_none(), "a link only: no warnings, no download");
        for a in [Action::InstallVersion("v0.2.0".into()), Action::RestartToUpdate] {
            assert_eq!(act(&mut st, &inp, a).error.as_deref(), Some(PORTABLE_REFUSED));
        }
    }

    /// A build from source lists only ("running from source: updates are off"), never checks by
    /// itself — the window's automatic check included — and refuses the actions; Check now still
    /// lists (here: the tests' offline network, its quiet line).
    #[test]
    fn dev_mode_says_updates_off() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = ctx();
        let mut st = UiState::new("h2", &scene);
        assert_eq!(st.data_mode, crate::paths::DataMode::Dev);
        st.update.background = true; // as the window sets it
        st.update.releases = releases();
        st.update.current = ver("0.1.0");
        st.update.open = true;
        let lines = shown(&ctx, &mut st, &inp);
        assert!(has(&lines, UPDATES_OFF), "{lines:#?}");
        assert!(!has(&lines, "Install and restart") && !has(&lines, "Download zip") && !has(&lines, "Restart now"));
        assert!(lines.iter().any(|l| l == "v0.2.0"), "the list is shown");
        assert!(!st.update.checking() && st.update.last_started.is_none(), "no automatic check from source");
        for a in [Action::InstallVersion("v0.2.0".into()), Action::RestartToUpdate] {
            assert_eq!(act(&mut st, &inp, a).error.as_deref(), Some(UPDATES_OFF));
        }
        act(&mut st, &inp, Action::CheckUpdates);
        let t = std::time::Instant::now();
        while st.update.checking() && t.elapsed().as_secs() < 10 {
            shown(&ctx, &mut st, &inp);
        }
        assert_eq!(st.update.error.as_deref(), Some("couldn't check (offline)"));
        assert_eq!(st.update.releases.len(), releases().len(), "the last list stays");
    }

    /// Each of the panel's buttons sends its action: Check for updates now, Install and restart,
    /// the download's Cancel ("Cancelling…" until the stalled read lets go), Restart now.
    ///
    /// Deterministic under load: the cancel is clicked only once the job's read is held at the
    /// gate (`stalled`), so the job can't report its cancel before the panel has drawn
    /// "Cancelling…" (it used to, when a busy machine left the job thread short of the installer's
    /// read at the click: it then saw the cancel between reads and ended at once); every wait
    /// polls its condition with a generous bound.
    #[test]
    fn buttons_emit_actions() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = ctx();
        let mut st = UiState::new("h2", &scene);
        let root = temp_root("buttons");
        let (gate, stalled) = installed_gated(&mut st, &root, "0.1.0");
        st.update.releases.clear();
        st.update.open = true;
        assert!(has(&shown(&ctx, &mut st, &inp), "No list yet"));
        let outs = click_widget(&ctx, &mut st, &inp, "check-now");
        assert_eq!(actions(&outs), [Action::CheckUpdates]);
        let t = std::time::Instant::now();
        while st.update.checking() && t.elapsed().as_secs() < 60 {
            shown(&ctx, &mut st, &inp);
        }
        assert_eq!(st.update.releases, releases(), "the list arrived");
        assert!(st.update.error.is_none());
        // a stalled download: its read held at the gate before anything else
        gate.store(true, Ordering::Relaxed);
        let outs = click_widget(&ctx, &mut st, &inp, "install-v0.2.0");
        assert_eq!(actions(&outs), [Action::InstallVersion("v0.2.0".into())]);
        assert!(wait_for(60, || stalled.load(Ordering::Relaxed)), "the job reached the installer's read");
        assert!(matches!(st.update.download.as_ref().unwrap().poll(), DownloadState::Running(..)));
        assert!(has(&shown(&ctx, &mut st, &inp), "downloading:"));
        let outs = click_widget(&ctx, &mut st, &inp, "cancel-download");
        assert_eq!(actions(&outs), [Action::CancelUpdateDownload]);
        assert!(st.update.download.is_none() && st.update.cancelling());
        assert!(has(&shown(&ctx, &mut st, &inp), "Cancelling…"), "until the stalled read lets go");
        gate.store(false, Ordering::Relaxed);
        let t = std::time::Instant::now();
        while st.update.cancelling() && t.elapsed().as_secs() < 60 {
            shown(&ctx, &mut st, &inp);
        }
        assert!(!st.update.cancelling() && !has(&shown(&ctx, &mut st, &inp), "Cancelling…"));
        // (the job removes its .part as it ends, just after reporting)
        let parts = || root.join("updates").read_dir().unwrap().flatten().any(|e| e.file_name().to_string_lossy().ends_with(".part"));
        assert!(wait_for(60, || !parts()), "the cancelled .part is gone");
        // again, to the end: Restart now
        click_widget(&ctx, &mut st, &inp, "install-v0.2.0");
        assert!(matches!(wait_download(&mut st), DownloadState::Ready(_)));
        let lines = shown(&ctx, &mut st, &inp);
        assert!(has(&lines, "Restart now") && has(&lines, "downloaded and verified"), "{lines:#?}");
        assert!(!has(&lines, "installer will ask"), "the installer never closes another window: {lines:#?}");
        let outs = click_widget(&ctx, &mut st, &inp, "restart-now");
        assert_eq!(actions(&outs), [Action::RestartToUpdate]);
        assert!(outs.iter().any(|o| o.restart_for_update), "nothing running: it goes ahead");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// One download at a time: starting another release's cancels the first ("Cancelling…" while
    /// it finishes its read); the same release again is left as it is.
    #[test]
    fn starting_another_download_cancels_the_first() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let mut st = UiState::new("h2", &scene);
        let root = temp_root("one-at-a-time");
        let gate = installed(&mut st, &root, "0.1.0");
        // a v0.2.1 like v0.2.0, its assets at its own addresses
        let mut next = st.update.releases.iter().find(|r| r.tag == "v0.2.0").unwrap().clone();
        (next.tag, next.version) = ("v0.2.1".into(), ver("0.2.1"));
        for a in &mut next.assets {
            a.name = a.name.replace("0.2.0", "0.2.1");
            a.url = crate::update::net::asset_url("v0.2.1", &a.name);
        }
        st.update.releases.insert(0, next);
        gate.store(true, Ordering::Relaxed);
        act(&mut st, &inp, Action::InstallVersion("v0.2.0".into()));
        act(&mut st, &inp, Action::InstallVersion("v0.2.0".into()));
        assert!(!st.update.cancelling(), "the same release again: left running");
        act(&mut st, &inp, Action::InstallVersion("v0.2.1".into()));
        assert_eq!(st.update.download.as_ref().map(|d| d.tag()), Some("v0.2.1"));
        assert!(st.update.cancelling(), "v0.2.0 is cancelled, finishing its read");
        gate.store(false, Ordering::Relaxed);
        // v0.2.1's stand-in isn't in the fixture sums: verification refuses it (nothing installed)
        assert!(matches!(wait_download(&mut st), DownloadState::Failed(e) if e.contains("SHA256SUMS")), "fails closed");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The window's automatic check (installed build, Check automatically on): at launch, then
    /// not again until 6 hours have passed; off with the switch. A refused check keeps the last
    /// list; a redirect the client refused is said as it is.
    #[test]
    fn the_automatic_check_runs_at_launch_then_every_six_hours() {
        let scene = Scene::preset("h2").unwrap();
        let cam = camera();
        let inp = input(&scene, None, &cam);
        let ctx = ctx();
        let mut st = UiState::new("h2", &scene);
        let root = temp_root("auto");
        installed(&mut st, &root, "0.1.0");
        let net = FixtureNet::new();
        let calls = net.calls.clone();
        st.update.net = Arc::new(net);
        st.update.releases.clear();
        shown(&ctx, &mut st, &inp);
        assert_eq!(calls.load(Ordering::Relaxed), 0, "headless (no window): no automatic check");
        st.update.background = true;
        let wait = |st: &mut UiState, ctx: &egui::Context| {
            let t = std::time::Instant::now();
            while st.update.checking() && t.elapsed().as_secs() < 10 {
                shown(ctx, st, &inp);
            }
        };
        shown(&ctx, &mut st, &inp);
        wait(&mut st, &ctx);
        assert_eq!(calls.load(Ordering::Relaxed), 1, "at launch");
        assert_eq!(st.update.releases.len(), 3);
        shown(&ctx, &mut st, &inp);
        assert_eq!(calls.load(Ordering::Relaxed), 1, "not again within 6 hours");
        let Some(earlier) = std::time::Instant::now().checked_sub(version::CHECK_EVERY) else { return };
        st.update.last_started = Some(earlier);
        shown(&ctx, &mut st, &inp);
        wait(&mut st, &ctx);
        assert_eq!(calls.load(Ordering::Relaxed), 2, "6 hours on");
        st.update.auto_check = false;
        st.update.last_started = Some(earlier);
        shown(&ctx, &mut st, &inp);
        assert_eq!(calls.load(Ordering::Relaxed), 2, "Check automatically off");
        // refusals: one quiet line, the last list kept
        st.update.open = true;
        for (e, line) in [
            (CheckError::RateLimited, "couldn't check (GitHub said: rate limited, try later)".to_string()),
            (CheckError::Malformed("GitHub's releases list redirected to an unexpected host: evil.example".into()), "couldn't check: GitHub's releases list redirected to an unexpected host: evil.example".to_string()),
            (CheckError::Malformed("not JSON (x)".into()), "couldn't check (GitHub's answer wasn't a releases list: not JSON (x))".to_string()),
        ] {
            assert_eq!(version::check_error_line(&e), line);
            st.update.net = Arc::new(FixtureNet { list: Err(e), ..FixtureNet::new() });
            act(&mut st, &inp, Action::CheckUpdates);
            wait(&mut st, &ctx);
            assert_eq!(st.update.error.as_deref(), Some(line.as_str()));
            assert_eq!(st.update.releases.len(), 3, "the last list stays");
            assert!(has(&shown(&ctx, &mut st, &inp), &line), "{line}");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The channel and Check automatically are kept in the profile (a profile without them gives
    /// Stable and on).
    #[test]
    fn channel_and_auto_check_persist_in_profile() {
        let scene = Scene::preset("h2").unwrap();
        let mut st = UiState::new("h2", &scene);
        assert_eq!((st.update.channel, st.update.auto_check), (Channel::Stable, true), "the defaults");
        let text = profile::capture(&st);
        assert!(text.contains("update_channel = Stable\n") && text.contains("update_auto_check = true\n"), "{text}");
        (st.update.channel, st.update.auto_check) = (Channel::Beta, false);
        let text = profile::capture(&st);
        let mut fresh = UiState::new("h2", &scene);
        profile::apply(&mut fresh, &text);
        assert_eq!((fresh.update.channel, fresh.update.auto_check), (Channel::Beta, false));
        // an older profile without the lines, and unreadable values: the defaults
        let mut old = UiState::new("h2", &scene);
        profile::apply(&mut old, "update_channel = Nightly\nupdate_auto_check = maybe\n");
        assert_eq!((old.update.channel, old.update.auto_check), (Channel::Stable, true));
        let defaults = profile::defaults(&old);
        profile::apply(&mut old, &defaults);
        assert_eq!((old.update.channel, old.update.auto_check), (Channel::Stable, true));
    }
}
