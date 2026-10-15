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
//! copy. The input replayer (`--replay … --screenshot`, `crate::replay`) shares the headless frame
//! ([`Shot`]) and the first-solve wait.
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
/// thermostat on before playing, `opacity=<x>` sets the volume's opacity (View ▸ Density), `wait=sampling` shoots while the estimated solver's first
/// sampling pass runs (instead of after the first solve), `drag=<i>` poses a Select-drag of nucleus
/// i (its pull label by the `cursor`), `pull=<Ha/Bohr>` sets the drag pull (as the wheel would),
/// `ptable=<query>` opens the periodic table with that search text (empty: none),
/// `names=common|stoichiometric|hydride|formula` sets the molecule name style,
/// `wait=bonds` shoots once the bond graph is in (the free-atom references solved: ≈ 70 s for
/// O at 57³ on the GPU).
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
    pub opacity: Option<f32>,
    pub wait_sampling: bool,
    pub drag: Option<usize>,
    pub pull: Option<f64>,
    pub ptable: Option<String>,
    pub wait_bonds: bool,
    pub names: Option<crate::names::NameStyle>,
    /// EXPERIMENTAL (dots / orbital preview shots): camera yaw / pitch (radians) and distance
    /// (Bohr), `render=volume|iso|dots`, the iso level, the volume gamma, `aids=off`,
    /// `dotcolour=orbital`, `nuclei=<scale>`.
    pub yaw: Option<f32>,
    pub pitch: Option<f32>,
    pub dist: Option<f32>,
    pub render: Option<u32>,
    pub iso: Option<f32>,
    pub gamma: Option<f32>,
    pub aids_off: bool,
    pub dots_by_orbital: bool,
    pub nucleus_scale: Option<f32>,
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
                "opacity" => s.opacity = Some(v.parse().map_err(|_| format!("--ui: bad value in {kv}"))?),
                "wait" => match v {
                    "sampling" => s.wait_sampling = true,
                    "bonds" => s.wait_bonds = true,
                    other => return Err(format!("--ui: wait=sampling|bonds, got {other}")),
                },
                "ptable" => s.ptable = Some(v.to_string()),
                "names" => {
                    use crate::names::NameStyle::*;
                    s.names = Some(match v {
                        "common" => Common,
                        "stoichiometric" => Stoichiometric,
                        "hydride" => Hydride,
                        "formula" => FormulaOnly,
                        other => return Err(format!("--ui: names=common|stoichiometric|hydride|formula, got {other}")),
                    })
                }
                "drag" => s.drag = Some(v.parse().map_err(|_| format!("--ui: bad value in {kv}"))?),
                "pull" => s.pull = Some(v.parse().map_err(|_| format!("--ui: bad value in {kv}"))?),
                "yaw" => s.yaw = Some(v.parse().map_err(|_| format!("--ui: bad value in {kv}"))?),
                "pitch" => s.pitch = Some(v.parse().map_err(|_| format!("--ui: bad value in {kv}"))?),
                "dist" => s.dist = Some(v.parse().map_err(|_| format!("--ui: bad value in {kv}"))?),
                "iso" => s.iso = Some(v.parse().map_err(|_| format!("--ui: bad value in {kv}"))?),
                "gamma" => s.gamma = Some(v.parse().map_err(|_| format!("--ui: bad value in {kv}"))?),
                "aids" => s.aids_off = v == "off",
                "dotcolour" => s.dots_by_orbital = v == "orbital",
                "nuclei" => s.nucleus_scale = Some(v.parse().map_err(|_| format!("--ui: bad value in {kv}"))?),
                "render" => {
                    s.render = Some(match v {
                        "volume" => 0,
                        "iso" => 1,
                        "dots" => 2,
                        _ => return Err(format!("--ui: render=volume|iso|dots, got {v}")),
                    })
                }
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
        if let Some(o) = self.opacity {
            viewer.ui.opacity = o;
        }
        if let Some(x) = self.yaw {
            viewer.camera.yaw = x;
        }
        if let Some(x) = self.pitch {
            viewer.camera.pitch = x;
        }
        if let Some(x) = self.dist {
            viewer.camera.distance = x;
        }
        if let Some(x) = self.render {
            viewer.ui.volume_mode = x;
        }
        if let Some(x) = self.iso {
            viewer.ui.iso = x;
        }
        if let Some(x) = self.gamma {
            viewer.ui.gamma = x;
        }
        viewer.ui.dots_by_orbital = self.dots_by_orbital;
        viewer.ui.nucleus_scale = self.nucleus_scale;
        if self.aids_off {
            viewer.ui.show_cell = false;
            viewer.ui.show_floor = false;
        }
        if let Some(c) = self.cursor {
            viewer.tool.cursor = Some(c);
        }
        if let Some(i) = self.drag {
            viewer.tool.selected = Some(i);
            viewer.tool.dragging = true;
        }
        if let Some(f) = self.pull {
            viewer.ui.queued.push(crate::ui::Action::DragForce(f));
        }
        if let Some(q) = &self.search {
            viewer.ui.search.toggle();
            viewer.ui.search.query = q.clone();
        }
        if let Some(m) = &self.menu {
            crate::ui::open_menu(&viewer.egui_ctx, m).ok_or_else(|| format!("--ui: unknown menu {m}"))?;
        }
        if let Some(n) = self.names {
            viewer.ui.name_style = n;
        }
        if let Some(q) = &self.ptable {
            viewer.ui.ptable_query = q.clone();
            crate::ui::open_ptable(&viewer.egui_ctx);
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

/// The headless frame `--screenshot` and `--replay … --screenshot` draw into: the harness's device
/// with an sRGB colour target (the window's surface format, so the PNG matches it), its depth,
/// and the painter.
pub struct Shot {
    pub hl: Headless,
    pub painter: Painter,
}

impl Shot {
    /// `None` without a GPU adapter.
    pub fn new(w: u32, h: u32) -> Option<Shot> {
        let mut hl = Headless::new(w, h)?;
        // Swap the harness's linear colour target for an sRGB one; its readback copies raw bytes.
        (hl.colour, hl.colour_view) = colour_target(&hl.ctx, w, h);
        let painter = Painter::new(&hl.ctx, FORMAT);
        Some(Shot { hl, painter })
    }

    /// New colour and depth targets at `w × h` (a replayed window resize), the painter rebound
    /// to the depth as the window's resize does.
    pub fn resize(&mut self, w: u32, h: u32) {
        if (w, h) == (self.hl.width, self.hl.height) || w == 0 || h == 0 {
            return;
        }
        let ctx = &self.hl.ctx;
        (self.hl.colour, self.hl.colour_view) = colour_target(ctx, w, h);
        self.hl.depth = ctx
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("screenshot depth"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: gpu::DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        (self.hl.width, self.hl.height) = (w, h);
        self.painter.rebind(ctx, &self.hl.depth);
    }

    /// Renders the viewer's frame — spheres, density and the whole chrome — through the window's
    /// code. `textures`: egui texture updates of frames run without painting (the replay's), applied
    /// first. A few frames: egui lays some widgets out over more than one pass (and a popup's first
    /// frame is an invisible sizing pass, then fades in over a few 1/60 s frames).
    pub fn render(&mut self, viewer: &mut Viewer, mut textures: egui::TexturesDelta) {
        let (w, h) = (self.hl.width, self.hl.height);
        for _ in 0..16 {
            let raw = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32, h as f32))),
                ..Default::default()
            };
            let mut out = viewer.run_ui(raw);
            textures.append(std::mem::take(&mut out.textures_delta));
            out.textures_delta = std::mem::take(&mut textures);
            self.painter.set_view(&self.hl.ctx, &viewer.ui, viewer.density.fade(), viewer.density_mix(Instant::now()));
            let target = Target {
                colour: &self.hl.colour_view,
                depth: &self.hl.depth,
                size: [w, h],
            };
            let shown = viewer.shown_scene(Instant::now());
            let lines = viewer.overlay_lines(&shown);
            let risk = viewer.atom_risk();
            let world = World {
                camera: &viewer.camera,
                nuclei: &shown.nuclei,
                selected: viewer.tool.selected,
                lines: &lines,
                viewport: viewer.viewport,
                risk: &risk,
            };
            self.painter.paint(&self.hl.ctx, target, world, &viewer.egui_ctx, out);
        }
    }

    /// Reads the frame back and writes it as a PNG.
    pub fn write(&self, path: &str) -> Result<(), String> {
        let (w, h) = (self.hl.width, self.hl.height);
        write_png(path, w, h, &self.hl.read_rgba())?;
        eprintln!("wrote {path} ({w}×{h})");
        Ok(())
    }
}

/// An sRGB colour target the readback can copy.
fn colour_target(ctx: &gpu::GpuContext, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let colour = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("screenshot colour"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = colour.create_view(&Default::default());
    (colour, view)
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
    let mut shot = Shot::new(w, h).ok_or("no GPU adapter")?;
    viewer.start(backend, Some(shot.hl.ctx.clone()));
    let (hl, painter) = (&shot.hl, &mut shot.painter);
    if o.ui.wait_sampling {
        wait_for_sampling(&mut viewer, hl, painter)?;
    } else {
        wait_for_first_solve(&mut viewer, hl, painter)?;
    }
    if sample {
        wait_for_model(&mut viewer, hl, painter)?;
    }
    if o.ui.wait_bonds {
        wait_for_bonds(&mut viewer, hl, painter)?;
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
                viewer.on_event(ev, &hl.ctx, painter, &hl.depth);
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
    shot.render(&mut viewer, Default::default());
    shot.write(path)
}

/// Blocks until the snapshot carries a bond graph with every free-atom reference in (`wait=bonds`).
fn wait_for_bonds(viewer: &mut Viewer, hl: &Headless, painter: &mut Painter) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs(600);
    loop {
        if viewer.snapshot.as_ref().is_some_and(|s| s.dynamics.bonds.is_some() && s.dynamics.bonds_pending.is_empty()) {
            return Ok(());
        }
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err("no bond graph within 600 s".into());
        }
        if let Some(ev) = viewer.sim.as_ref().ok_or("no worker")?.recv_timeout(left.min(Duration::from_secs(1))) {
            viewer.on_event(ev, &hl.ctx, painter, &hl.depth);
        }
    }
}

/// Blocks until the worker finishes its first solve (the Busy(false) after it).
pub fn wait_for_first_solve(
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
