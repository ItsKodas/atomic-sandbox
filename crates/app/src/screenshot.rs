//! Hidden `--screenshot <path> [--preset <name>] [--backend gpu|cpu] [--play-ms <n>]` mode:
//! solves the preset, optionally plays the dynamics for n ms of wall time (the frame then shows
//! the trajectory mid-flight: playing, readouts and sparkline), then renders one 1280×800 frame
//! (spheres, density and the panel) headlessly through the same `Painter` and `Viewer` code as
//! the window, and writes it as a PNG. For docs and for checking the viewer without driving a
//! window. The GPU backend shares the headless device, as the window's does, so its density
//! texture is bound without a copy.
use crate::gfx::{Painter, Target, World};
use crate::sim::{Backend, Command, Event as SimEvent};
use crate::viewer::Viewer;
use render::headless::Headless;
use std::time::{Duration, Instant};

const W: u32 = 1280;
const H: u32 = 800;
/// The window surface is sRGB; render the screenshot the same way so the PNG matches it.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

pub fn run(path: &str, preset: &str, backend: Backend, play_ms: Option<u64>) -> Result<(), String> {
    let mut viewer = Viewer::new(preset).ok_or_else(|| format!("unknown preset {preset}"))?;
    let mut hl = Headless::new(W, H).ok_or("no GPU adapter")?;
    viewer.start(backend, Some(hl.ctx.clone()));
    // Swap the harness's linear colour target for an sRGB one; its readback copies raw bytes.
    let size = wgpu::Extent3d {
        width: W,
        height: H,
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
    wait_for_first_solve(&mut viewer, &hl, &mut painter)?;
    if let Some(ms) = play_ms {
        viewer.send(Command::Dyn(sim::DynCommand::Play));
        let end = Instant::now() + Duration::from_millis(ms);
        while Instant::now() < end {
            let left = end.saturating_duration_since(Instant::now());
            if let Some(ev) = viewer.sim.as_ref().ok_or("no worker")?.recv_timeout(left) {
                viewer.on_event(ev, &hl.ctx, &mut painter, &hl.depth);
            }
        }
        if let Some(e) = viewer.snapshot.as_ref().and_then(|s| s.dynamics.last_error.clone()) {
            return Err(format!("the run paused on a solver failure: {e}"));
        }
    }
    // A few frames: egui lays some widgets out over more than one pass.
    for _ in 0..3 {
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(W as f32, H as f32),
            )),
            ..Default::default()
        };
        let out = viewer.run_ui(raw);
        painter.set_view(&hl.ctx, &viewer.ui);
        let target = Target {
            colour: &hl.colour_view,
            depth: &hl.depth,
            size: [W, H],
        };
        let lines = crate::overlay::build(&viewer.ui, &viewer.scene, &viewer.camera, viewer.ui.mode);
        let world = World {
            camera: &viewer.camera,
            nuclei: &viewer.scene.nuclei,
            selected: viewer.tool.selected,
            lines: &lines,
        };
        painter.paint(&hl.ctx, target, world, &viewer.egui_ctx, out);
    }
    write_png(path, &hl.read_rgba())?;
    eprintln!("wrote {path} ({W}×{H})");
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

fn write_png(path: &str, px: &[[u8; 4]]) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("{path}: {e}"))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), W, H);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let bytes: Vec<u8> = px.iter().flat_map(|p| [p[0], p[1], p[2], 255]).collect();
    let mut w = enc.write_header().map_err(|e| e.to_string())?;
    w.write_image_data(&bytes).map_err(|e| e.to_string())
}
