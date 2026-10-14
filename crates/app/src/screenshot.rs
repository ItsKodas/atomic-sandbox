//! Hidden `--screenshot <path> [--preset <name>] [--backend gpu|cpu|estimated] [--estimated]
//! [--sample] [--audit <n>] [--play-ms <n>] [--size <w>x<h>] [--ui "<k=v;…>"]` mode: solves the
//! preset (`--estimated`: the surrogate taught by the GPU solver; its sampling pass starts at
//! once, and `--sample` waits for it to finish and for a trusted model before anything else),
//! optionally plays the dynamics for n ms of wall time (the frame then shows the trajectory
//! mid-flight: playing, readouts and sparkline — coloured by source when estimated), puts the
//! chrome into the `--ui` state ([`ShotUi`]: an open menu, a dock tab, a selection, the tool…),
//! then renders one frame (1400×900 by default, the window's size: spheres, density and the
//! whole chrome) headlessly through the same `Painter` and `Viewer` code as the window, and writes
//! it as a PNG. For docs and for checking the viewer without driving a window. The GPU backend
//! shares the headless device, as the window's does, so its density texture is bound without a
//! copy.
use crate::gfx::{Painter, Target, World};
use crate::sim::{Backend, Command, Event as SimEvent};
use crate::viewer::Viewer;
use render::headless::Headless;
use std::time::{Duration, Instant};

/// The default frame: the window's default size.
const W: u32 = 1400;
const H: u32 = 900;
/// The window surface is sRGB; render the screenshot the same way so the PNG matches it.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// The chrome's state for the frame (`--ui "k=v;k=v"`, for the docs' screenshots of each state
/// of the mockup): `menu=file|scene|view|physics|help` opens a menu, `tab=readouts|graph|model|
/// selection|log` picks the dock tab, `dock=closed` collapses it, `select=<i>` selects a nucleus,
/// `tool=orbit|place|select`, `element=<Z>`, `cursor=<x>,<y>` puts the pointer over the view
/// (Place mode's guides), `search=<text>` opens Ctrl K with a query, `thermostat=<K>` switches the
/// thermostat on before playing, `wait=sampling` shoots while the estimated solver's first
/// sampling pass runs (instead of after the first solve).
#[derive(Default)]
pub struct ShotUi {
    pub menu: Option<String>,
    pub tab: Option<String>,
    pub dock_closed: bool,
    pub select: Option<usize>,
    pub tool: Option<String>,
    pub element: Option<u8>,
    pub cursor: Option<(f32, f32)>,
    pub search: Option<String>,
    pub thermostat: Option<f64>,
    pub wait_sampling: bool,
}

impl ShotUi {
    pub fn parse(spec: &str) -> Result<ShotUi, String> {
        let mut s = ShotUi::default();
        for kv in spec.split(';').filter(|x| !x.trim().is_empty()) {
            let (k, v) = kv.split_once('=').ok_or_else(|| format!("--ui: {kv} is not key=value"))?;
            match k.trim() {
                "menu" => s.menu = Some(v.to_string()),
                "tab" => s.tab = Some(v.to_string()),
                "dock" => s.dock_closed = v == "closed",
                "select" => s.select = Some(v.parse().map_err(|_| format!("--ui: bad value in {kv}"))?),
                "tool" => s.tool = Some(v.to_string()),
                "element" => s.element = Some(v.parse().map_err(|_| format!("--ui: bad value in {kv}"))?),
                "cursor" => {
                    let (x, y) = v.split_once(',').ok_or_else(|| format!("--ui: cursor=x,y, got {v}"))?;
                    s.cursor = Some((x.parse().map_err(|_| format!("--ui: bad value in {kv}"))?, y.parse().map_err(|_| format!("--ui: bad value in {kv}"))?));
                }
                "search" => s.search = Some(v.to_string()),
                "thermostat" => s.thermostat = Some(v.parse().map_err(|_| format!("--ui: bad value in {kv}"))?),
                "wait" => s.wait_sampling = v == "sampling",
                other => return Err(format!("--ui: unknown key {other}")),
            }
        }
        Ok(s)
    }

    /// Puts the chrome into the requested state (before the frames are rendered).
    fn apply(&self, viewer: &mut Viewer) -> Result<(), String> {
        use crate::ui::{DockTab, ToolMode};
        if let Some(t) = &self.tab {
            viewer.ui.dock_tab = match t.as_str() {
                "readouts" => DockTab::Readouts,
                "graph" => DockTab::Graph,
                "model" => DockTab::Model,
                "selection" => DockTab::Selection,
                "log" => DockTab::Log,
                other => return Err(format!("--ui: unknown tab {other}")),
            };
        }
        viewer.ui.dock_open = !self.dock_closed;
        if let Some(i) = self.select {
            viewer.tool.selected = Some(i);
        }
        if let Some(t) = &self.tool {
            viewer.ui.mode = match t.as_str() {
                "orbit" => ToolMode::Orbit,
                "place" => ToolMode::Place,
                "select" => ToolMode::Select,
                other => return Err(format!("--ui: unknown tool {other}")),
            };
        }
        if let Some(z) = self.element {
            viewer.ui.element = z;
        }
        if let Some(c) = self.cursor {
            viewer.tool.cursor = Some(c);
        }
        if let Some(q) = &self.search {
            viewer.ui.search.toggle();
            viewer.ui.search.query = q.clone();
        }
        if let Some(m) = &self.menu {
            crate::ui::open_menu(&viewer.egui_ctx, m).ok_or_else(|| format!("--ui: unknown menu {m}"))?;
        }
        Ok(())
    }
}

/// Everything the hidden flag takes besides the output path.
pub struct ShotOptions {
    pub preset: String,
    pub backend: Backend,
    pub play_ms: Option<u64>,
    pub sample: bool,
    pub audit: Option<usize>,
    pub size: Option<(u32, u32)>,
    pub ui: ShotUi,
}

/// Renders the shot; `setup` applies the start-up flags (`--scene`, `--walls`, `--draft`,
/// `--cloud-refresh`) to the viewer before its worker starts.
pub fn run(path: &str, o: &ShotOptions, setup: &dyn Fn(&mut Viewer) -> Result<(), String>) -> Result<(), String> {
    let (preset, backend, play_ms, sample, audit) = (o.preset.as_str(), o.backend, o.play_ms, o.sample, o.audit);
    let (w, h) = o.size.unwrap_or((W, H));
    let mut viewer = Viewer::new(preset).ok_or_else(|| format!("unknown preset {preset}"))?;
    setup(&mut viewer)?;
    if let Some(n) = audit {
        viewer.ui.audit_every = n;
    }
    let mut hl = Headless::new(w, h).ok_or("no GPU adapter")?;
    viewer.start(backend, Some(hl.ctx.clone()));
    // Swap the harness's linear colour target for an sRGB one; its readback copies raw bytes.
    let size = wgpu::Extent3d {
        width: w,
        height: h,
        depth_or_array_layers: 1,
    };
    hl.colour = hl.ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("screenshot colour"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    hl.colour_view = hl.colour.create_view(&Default::default());
    let mut painter = Painter::new(&hl.ctx, FORMAT);
    if o.ui.wait_sampling {
        wait_for_sampling(&mut viewer, &hl, &mut painter)?;
    } else {
        wait_for_first_solve(&mut viewer, &hl, &mut painter)?;
    }
    if sample {
        wait_for_model(&mut viewer, &hl, &mut painter)?;
    }
    if let Some(t) = o.ui.thermostat {
        viewer.ui.thermostat_on = true;
        viewer.ui.thermostat_t = t;
        viewer.send(Command::Dyn(sim::DynCommand::SetThermostat(Some((t, viewer.ui.thermostat_tau)))));
    }
    if let Some(ms) = play_ms.filter(|_| !o.ui.wait_sampling) {
        if backend == Backend::Estimated {
            // the surrogate answers in microseconds: run the full steps per tick
            viewer.ui.speed = sim::dynamics::MAX_SPEED;
            viewer.send(Command::Dyn(sim::DynCommand::SetSpeed(viewer.ui.speed)));
        }
        viewer.send(Command::Dyn(sim::DynCommand::Play));
        let end = Instant::now() + Duration::from_millis(ms);
        // what the play did, for checking the viewer extras: the farthest coordinate, the
        // extremes of E conserved, side densities and draft-mode banners seen
        let (mut far, mut e_lo, mut e_hi, mut sides, mut banners) = (0.0f64, f64::INFINITY, f64::NEG_INFINITY, 0u32, 0u32);
        let mut last_side = None;
        while Instant::now() < end {
            let left = end.saturating_duration_since(Instant::now());
            if let Some(ev) = viewer.sim.as_ref().ok_or("no worker")?.recv_timeout(left) {
                viewer.on_event(ev, &hl.ctx, &mut painter, &hl.depth);
                if let Some(s) = viewer.snapshot.as_ref().filter(|s| !s.dynamics.paused) {
                    let d = &s.dynamics;
                    far = d.scene.nuclei.iter().flat_map(|n| n.pos).fold(far, |m, x| m.max(x.abs()));
                    if !d.history.is_empty() {
                        e_lo = e_lo.min(d.energy_conserved);
                        e_hi = e_hi.max(d.energy_conserved);
                    }
                    if let Some(sd) = d.side_display.as_ref().filter(|sd| last_side != Some(sd.seq)) {
                        last_side = Some(sd.seq);
                        sides += 1;
                    }
                    banners += u32::from(viewer.draft_banner().is_some());
                }
            }
        }
        if let Some(s) = &viewer.snapshot {
            let d = &s.dynamics;
            eprintln!(
                "play summary: t {:.1} fs, step {}, farthest |coordinate| {far:.3} Bohr (walls {:?}, cell max {:.3}), E conserved spread {:.2e} Ha, walls work {:+.2e} Ha, drift {:?} Ha/ps, side densities {sides}, draft banners on {banners} snapshots, profiles {}, density label {:?}, last error {:?}",
                d.sim_time_fs(),
                d.step,
                d.walls.map(|w| w.half_width),
                d.scene.cell.max_coord(),
                e_hi - e_lo,
                d.walls_work,
                d.drift_per_ps,
                s.atom_profiles.len(),
                viewer.density.label(d.step),
                d.last_error
            );
        }
        if let Some(e) = viewer.snapshot.as_ref().and_then(|s| s.dynamics.last_error.clone()) {
            return Err(format!("the run paused on a solver failure: {e}"));
        }
    }
    o.ui.apply(&mut viewer)?;
    // A few frames: egui lays some widgets out over more than one pass (and a popup's first
    // frame is an invisible sizing pass, then fades in over a few 1/60 s frames).
    for _ in 0..16 {
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(w as f32, h as f32),
            )),
            ..Default::default()
        };
        let out = viewer.run_ui(raw);
        painter.set_view(&hl.ctx, &viewer.ui, viewer.density.fade(), viewer.density_mix(Instant::now()));
        let target = Target {
            colour: &hl.colour_view,
            depth: &hl.depth,
            size: [w, h],
        };
        let lines = crate::overlay::build(&viewer.ui, &viewer.scene, &viewer.camera, viewer.ui.mode);
        let shown = viewer.shown_scene(Instant::now());
        let risk = viewer.atom_risk();
        let world = World {
            camera: &viewer.camera,
            nuclei: &shown.nuclei,
            selected: viewer.tool.selected,
            lines: &lines,
            viewport: viewer.viewport,
            risk: &risk,
        };
        painter.paint(&hl.ctx, target, world, &viewer.egui_ctx, out);
    }
    write_png(path, w, h, &hl.read_rgba())?;
    eprintln!("wrote {path} ({w}×{h})");
    Ok(())
}

/// Blocks until the worker finishes its first solve (the Busy(false) after it).
fn wait_for_first_solve(
    viewer: &mut Viewer,
    hl: &Headless,
    painter: &mut Painter,
) -> Result<(), String> {
    loop {
        let ev = viewer
            .sim
            .as_ref()
            .ok_or("no worker")?
            .recv_timeout(Duration::from_secs(600))
            .ok_or("no solve within 600 s")?;
        let done = matches!(ev, SimEvent::Busy(false));
        viewer.on_event(ev, &hl.ctx, painter, &hl.depth);
        if done {
            return match &viewer.snapshot {
                Some(_) => Ok(()),
                None => Err(viewer
                    .last_error
                    .clone()
                    .unwrap_or_else(|| "no snapshot".into())),
            };
        }
    }
}

/// Blocks until the estimated solver's first sampling pass is running and has done a few jobs
/// (the sampling state of the mockup).
fn wait_for_sampling(viewer: &mut Viewer, hl: &Headless, painter: &mut Painter) -> Result<(), String> {
    loop {
        let running = viewer.snapshot.as_ref().is_some_and(|s| {
            crate::sim::sampling_active(&s.dynamics) && s.dynamics.sampling.as_ref().is_some_and(|p| p.done >= 8)
        });
        if running {
            return Ok(());
        }
        let ev = viewer
            .sim
            .as_ref()
            .ok_or("no worker")?
            .recv_timeout(Duration::from_secs(600))
            .ok_or("no sampling pass within 600 s")?;
        viewer.on_event(ev, &hl.ctx, painter, &hl.depth);
        if viewer.snapshot.as_ref().is_some_and(|s| s.backend != Backend::Estimated) {
            return Err("wait=sampling needs the estimated solver (--estimated)".into());
        }
        if viewer.snapshot.as_ref().is_some_and(|s| s.dynamics.sampling.as_ref().is_some_and(|p| p.finished)) {
            return Err("the pass finished before the shot".into());
        }
    }
}

/// Blocks until the estimated solver's sampling pass has ended and its model is trusted.
fn wait_for_model(viewer: &mut Viewer, hl: &Headless, painter: &mut Painter) -> Result<(), String> {
    let t0 = Instant::now();
    loop {
        let ready = viewer.snapshot.as_ref().is_some_and(|s| {
            !crate::sim::sampling_active(&s.dynamics) && s.model.as_ref().is_some_and(|m| m.fitted)
        });
        if ready {
            let s = viewer.snapshot.as_ref().expect("checked");
            let m = s.model.as_ref().expect("checked");
            eprintln!(
                "sampling pass done in {:.1} s: {} samples, trusted {}{}{}",
                t0.elapsed().as_secs_f64(),
                m.samples,
                m.trusted,
                m.untrusted_reason.as_deref().map_or(String::new(), |r| format!(" ({r})")),
                m.fit.map_or(String::new(), |(n, q, e, f)| format!("; fit on {n} (Q′ {q}): rmse_E {e:.2e} Ha/atom, rmse_F {f:.2e} Ha/Bohr"))
            );
            return if m.trusted { Ok(()) } else { Err("the model is not trusted after the pass".into()) };
        }
        let ev = viewer
            .sim
            .as_ref()
            .ok_or("no worker")?
            .recv_timeout(Duration::from_secs(600))
            .ok_or("no sampling progress within 600 s")?;
        viewer.on_event(ev, &hl.ctx, painter, &hl.depth);
        if viewer.snapshot.as_ref().is_some_and(|s| s.backend != Backend::Estimated) {
            return Err("--sample needs the estimated solver (--estimated)".into());
        }
    }
}

fn write_png(path: &str, w: u32, h: u32, px: &[[u8; 4]]) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("{path}: {e}"))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let bytes: Vec<u8> = px.iter().flat_map(|p| [p[0], p[1], p[2], 255]).collect();
    let mut w = enc.write_header().map_err(|e| e.to_string())?;
    w.write_image_data(&bytes).map_err(|e| e.to_string())
}
