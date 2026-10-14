//! GPU side of the viewer, independent of where the frame goes (window surface or the headless
//! screenshot target): density volume (upload or bind), volume look, sphere instances, and the
//! egui pass.
use crate::scene::SceneNucleus;
use crate::sim::Snapshot;
use crate::ui::UiState;
use gpu::{GpuContext, Volume3D};
use render::camera::{OrbitCamera, Viewport};
use render::elements;
use render::lines::LineVertex;
use render::spheres::SphereInstance;
use render::volume::VolumeUniform;
use render::Renderer;
use sim::Display;
use std::sync::Arc;

/// Background colour (linear).
const CLEAR: wgpu::Color = wgpu::Color {
    r: 0.012,
    g: 0.014,
    b: 0.022,
    a: 1.0,
};

/// Where one frame is drawn: a colour view of the painter's format and a matching Depth32Float view
/// (the one the volume pass is bound to).
pub struct Target<'a> {
    pub colour: &'a wgpu::TextureView,
    pub depth: &'a wgpu::TextureView,
    /// Physical pixels.
    pub size: [u32; 2],
}

/// What the 3D part of a frame shows.
pub struct World<'a> {
    pub camera: &'a OrbitCamera,
    pub nuclei: &'a [SceneNucleus],
    pub selected: Option<usize>,
    /// Reference geometry (cell, grids, axes) as line-list vertices; see `overlay::build`.
    pub lines: &'a [LineVertex],
    /// The rect of the target the scene is drawn into (the 3D view the chrome leaves); outside it
    /// the target keeps the clear colour, under the opaque egui panels.
    pub viewport: Viewport,
    /// Per nucleus, how far it is beyond what the model may answer with (draft mode:
    /// `viewer::Viewer::atom_risk`; empty = no tint): the sphere is mixed toward red by
    /// [`risk_tint`] of it.
    pub risk: &'a [f32],
}

pub struct Painter {
    pub renderer: Renderer,
    /// Upload targets for CPU densities, used in turn (the density crossfade keeps the previous
    /// one bound): created by the painter, never handed out (so uploading into them cannot touch
    /// a texture the GPU provider owns).
    uploads: [Option<Arc<Volume3D>>; 2],
    /// The CPU density last uploaded (into `shown`): a snapshot repeating it (a paused tick, a
    /// pin toggle) is not uploaded again.
    uploaded: Option<Arc<Vec<f32>>>,
    /// The texture the volume pass shows as the current density: an upload, or the texture of a
    /// GPU-backend snapshot. Holding it keeps the GPU provider's pool from reusing it.
    shown: Option<Arc<Volume3D>>,
    /// The density before it (the crossfade's other end; `None` after a snap), held likewise.
    previous: Option<Arc<Volume3D>>,
    /// `shown`'s grid: n, h and the origin (a change snaps the crossfade).
    shown_grid: Option<(usize, f64, [f64; 3])>,
    /// The last real (solved) density shown and its grid: the other end of the atomic
    /// superposition's blend ([`Painter::show_superposition`]).
    real: Option<(Arc<Volume3D>, (usize, f64, [f64; 3]))>,
    egui: egui_wgpu::Renderer,
}

/// What [`Painter::show_display`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shown {
    /// The density is the one already shown.
    Same,
    /// A new density, crossfaded from the previous one (same grid).
    Blended,
    /// A new density with nothing to fade from (the first, or a new grid): snapped.
    Snapped,
}

impl Painter {
    pub fn new(ctx: &GpuContext, format: wgpu::TextureFormat) -> Painter {
        let options = egui_wgpu::RendererOptions {
            msaa_samples: 1,
            ..Default::default()
        };
        Painter {
            renderer: Renderer::new(ctx, format),
            uploads: [None, None],
            uploaded: None,
            shown: None,
            previous: None,
            shown_grid: None,
            real: None,
            egui: egui_wgpu::Renderer::new(&ctx.device, format, options),
        }
    }

    /// Shows a snapshot's density (`DynSnapshot::display`, ruling R5) through
    /// [`Painter::show_display`]. A snapshot without a density leaves the last one bound,
    /// deliberately: a surrogate step (`Source::Surrogate`, M5) has no electrons, so the viewer
    /// keeps showing the last DFT cloud — faded through [`Painter::set_view`]'s `fade` and
    /// labelled by `viewer::DensityShown` — and so does a failed first evaluation (unfaded: there
    /// is nothing newer).
    pub fn show_snapshot(
        &mut self,
        ctx: &GpuContext,
        snap: &Snapshot,
        depth: &wgpu::TextureView,
    ) -> Result<Shown, String> {
        match &snap.dynamics.display {
            Some(d) => self.show_display(ctx, d, depth),
            None => Ok(Shown::Same),
        }
    }

    /// Shows a density: a CPU one is uploaded into the painter's own textures (the one not
    /// shown, so the previous density stays intact for the crossfade; recreated on a grid change;
    /// skipped when it is the density already uploaded), a GPU one's texture is bound as is — no
    /// copy. The volume pass is bound to the pair (previous, new) whenever the texture changes —
    /// the crossfade's `mix` comes from [`Painter::set_view`] — or to the new one alone on a grid
    /// change (snap); then the transfer function is auto-scaled to the new density's maximum.
    /// Holding both textures keeps the GPU provider's pool from reusing them. A density whose
    /// size does not match its grid is refused rather than shown.
    pub fn show_display(
        &mut self,
        ctx: &GpuContext,
        display: &Display,
        depth: &wgpu::TextureView,
    ) -> Result<Shown, String> {
        let (cell, origin, max) = match display {
            Display::Cpu { cell, origin, max, .. } | Display::Gpu { cell, origin, max, .. } => {
                (cell, origin, *max)
            }
        };
        let n = cell.n as u32;
        let vol = match display {
            Display::Cpu { rho, .. } => {
                if rho.len() != cell.n.pow(3) {
                    return Err(format!(
                        "snapshot density has {} values for a {}³ grid",
                        rho.len(),
                        cell.n
                    ));
                }
                match self.shown.clone() {
                    Some(v) if self.uploaded.as_ref().is_some_and(|u| Arc::ptr_eq(u, rho)) => v,
                    _ => {
                        // never into the texture on screen: it is the crossfade's other end
                        let keep = self.shown.clone();
                        let v = self.upload_free(ctx, rho, n, keep.as_ref());
                        self.uploaded = Some(rho.clone());
                        v
                    }
                }
            }
            Display::Gpu { volume, .. } if volume.dims == [n; 3] => {
                self.uploaded = None;
                volume.clone()
            }
            Display::Gpu { volume, .. } => {
                return Err(format!(
                    "snapshot texture is {:?} for a {}³ grid",
                    volume.dims, cell.n
                ))
            }
        };
        let grid = (cell.n, cell.h, *origin);
        let shown = if self.shown.as_ref().is_some_and(|s| Arc::ptr_eq(s, &vol)) {
            Shown::Same
        } else {
            let same_grid = self.shown_grid == Some(grid);
            self.previous = if same_grid { self.shown.take() } else { None };
            match &self.previous {
                Some(prev) => self.renderer.volume.bind_pair(ctx, prev, &vol, depth),
                None => self.renderer.volume.bind(ctx, &vol, depth),
            }
            if same_grid { Shown::Blended } else { Shown::Snapped }
        };
        self.real = Some((vol.clone(), grid));
        self.shown = Some(vol);
        self.shown_grid = Some(grid);
        self.set_box(ctx, cell, origin, max);
        Ok(shown)
    }

    /// Shows the atomic superposition (`crate::cloud`, a CPU density on the scene's grid) blended
    /// with the last real density: the pair (real, superposition) is bound, so `set_view`'s `mix`
    /// is the superposition's weight (`viewer::Viewer::density_mix`); without a real density on
    /// the same grid the superposition alone. Uploaded into the texture that does not hold the
    /// real density.
    pub fn show_superposition(&mut self, ctx: &GpuContext, display: &Display, depth: &wgpu::TextureView) -> Result<(), String> {
        let Display::Cpu { cell, origin, rho, max } = display else {
            return Err("the superposition is a CPU density".into());
        };
        let n = cell.n as u32;
        if rho.len() != cell.n.pow(3) {
            return Err(format!("superposition has {} values for a {}³ grid", rho.len(), cell.n));
        }
        let real = self.real.clone();
        // never into the real density's texture: it is the blend's other end
        let vol = self.upload_free(ctx, rho, n, real.as_ref().map(|(r, _)| r));
        self.uploaded = None;
        let grid = (cell.n, cell.h, *origin);
        self.previous = real.filter(|(_, g)| *g == grid).map(|(v, _)| v);
        match &self.previous {
            Some(r) => self.renderer.volume.bind_pair(ctx, r, &vol, depth),
            None => self.renderer.volume.bind(ctx, &vol, depth),
        }
        self.shown = Some(vol);
        self.shown_grid = Some(grid);
        self.set_box(ctx, cell, origin, *max);
        Ok(())
    }

    /// No density at all (the empty scene, `sim::dynamics` "The empty scene": no electrons): the
    /// volume pass draws nothing until the next density, which snaps in (nothing to fade from).
    /// Dropping the textures also returns the GPU provider's to its pool.
    pub fn clear_density(&mut self) {
        self.renderer.volume.unbind();
        self.shown = None;
        self.previous = None;
        self.shown_grid = None;
        self.real = None;
        self.uploaded = None;
    }

    /// Back to the last real density alone (the superposition switched off).
    pub fn show_real(&mut self, ctx: &GpuContext, depth: &wgpu::TextureView) {
        if let Some((v, grid)) = self.real.clone() {
            self.renderer.volume.bind(ctx, &v, depth);
            self.previous = None;
            self.shown = Some(v);
            self.shown_grid = Some(grid);
        }
    }

    /// Uploads `rho` into the upload texture that is not `keep` (created, or recreated for new
    /// dims) and returns it.
    fn upload_free(&mut self, ctx: &GpuContext, rho: &[f32], n: u32, keep: Option<&Arc<Volume3D>>) -> Arc<Volume3D> {
        let holds = |v: &Option<Arc<Volume3D>>| matches!((v, keep), (Some(v), Some(k)) if Arc::ptr_eq(v, k));
        let slot = if holds(&self.uploads[0]) { 1 } else { 0 };
        if self.uploads[slot].as_ref().is_none_or(|v| v.dims != [n; 3]) {
            self.uploads[slot] = Some(Arc::new(Volume3D::new(ctx, [n; 3])));
        }
        let vol = self.uploads[slot].clone().expect("created above");
        vol.upload(ctx, rho);
        vol
    }

    /// The volume pass's box and auto-scale for a density on `cell` with its first point at
    /// `origin` and maximum `max`.
    fn set_box(&mut self, ctx: &GpuContext, cell: &sim::Cell, origin: &[f64; 3], max: f32) {
        let (n, h) = (cell.n as u32, cell.h as f32);
        let lo = origin.map(|x| x as f32);
        let p = &mut self.renderer.volume_params;
        p.scale = max.max(1e-12);
        p.box_min = lo;
        p.box_max = lo.map(|x| x + (n - 1) as f32 * h);
        p.step = (h / 2.0).min(0.1);
        self.renderer.sync_volume_params(ctx);
    }

    /// After a resize the depth view is new: the volume pass must sample the new one.
    pub fn rebind(&mut self, ctx: &GpuContext, depth: &wgpu::TextureView) {
        if let Some(v) = &self.shown {
            match &self.previous {
                Some(p) => self.renderer.volume.bind_pair(ctx, p, v, depth),
                None => self.renderer.volume.bind(ctx, v, depth),
            }
        }
    }

    /// Copies the panel's look settings, the density's `fade` (1 = current, less = the last
    /// DFT density behind a surrogate step: `viewer::DensityShown::fade`) and the crossfade's
    /// `mix` (0 = the previous density, 1 = the current: `viewer::Viewer::density_mix`; 1 when
    /// there is nothing to fade from) into the volume uniform; uploads only when they changed.
    pub fn set_view(&mut self, ctx: &GpuContext, ui: &UiState, fade: f32, mix: f32) {
        let old = self.renderer.volume_params;
        let new = VolumeUniform {
            mode: ui.volume_mode,
            gamma: ui.gamma,
            opacity: ui.opacity,
            iso: ui.iso,
            fade,
            mix: if self.previous.is_some() { mix.clamp(0.0, 1.0) } else { 1.0 },
            ..old
        };
        if new != old {
            self.renderer.volume_params = new;
            self.renderer.sync_volume_params(ctx);
        }
    }

    /// Draws spheres + density, then the egui output on top, and submits.
    pub fn paint(
        &mut self,
        ctx: &GpuContext,
        target: Target<'_>,
        world: World<'_>,
        egui_ctx: &egui::Context,
        out: egui::FullOutput,
    ) {
        let [w, h] = target.size;
        // the scene in its viewport (clamped to the target): aspect, pixel → NDC, viewport and scissor
        let vp = world.viewport;
        let vp = if vp.size[0] < 1.0 || vp.size[1] < 1.0 { Viewport::full(w as f32, h as f32) } else { vp };
        self.renderer.update_camera_in(ctx, world.camera, vp);
        self.renderer
            .spheres
            .set_instances(ctx, &instances(world.nuclei, world.selected, world.risk));
        self.renderer.lines.set_lines(ctx, world.lines);
        for (id, deltas) in &out.textures_delta.set {
            for delta in deltas {
                self.egui
                    .update_texture(&ctx.device, &ctx.queue, *id, delta);
            }
        }
        let jobs = egui_ctx.tessellate(out.shapes, out.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: target.size,
            pixels_per_point: out.pixels_per_point,
        };
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        self.renderer
            .render_in(&mut enc, target.colour, target.depth, CLEAR, Some((vp, target.size)));
        let extra = self
            .egui
            .update_buffers(&ctx.device, &ctx.queue, &mut enc, &jobs, &screen);
        {
            let mut pass = render::volume::overlay_pass(&mut enc, target.colour).forget_lifetime();
            self.egui.render(&mut pass, &jobs, &screen);
        }
        ctx.queue.submit(extra.into_iter().chain([enc.finish()]));
        for id in &out.textures_delta.free {
            self.egui.free_texture(id);
        }
    }
}

/// Draft mode's tint: the fraction of red mixed into a sphere at `risk` (the larger of its force
/// uncertainty ÷ bound and its envelope distance ÷ limit): none at or below half the bound,
/// growing to 0.85 at twice it (capped, so the element colour still shows).
pub fn risk_tint(risk: f32) -> f32 {
    ((risk - 0.5) / 1.5).clamp(0.0, 1.0) * 0.85
}

const RISK_RED: [f32; 3] = [1.0, 0.12, 0.08];

fn instances(nuclei: &[SceneNucleus], selected: Option<usize>, risk: &[f32]) -> Vec<SphereInstance> {
    nuclei
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let base = elements::colour(n.z);
            let t = risk.get(i).map_or(0.0, |&r| risk_tint(r));
            SphereInstance {
                centre: n.pos.map(|x| x as f32),
                radius: elements::radius(n.z),
                colour: std::array::from_fn(|k| base[k] + t * (RISK_RED[k] - base[k])),
                selected: if selected == Some(i) { 1.0 } else { 0.0 },
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use render::camera::OrbitCamera;
    use render::headless::{Headless, HEADLESS_FORMAT};

    /// A CPU display of a Gaussian (width 1 Bohr) at `centre` on an n³ grid of spacing 0.25.
    fn gaussian(n: usize, centre: [f64; 3]) -> Display {
        let h = 0.25;
        let lo = -(n as f64 - 1.0) * h / 2.0;
        let mut rho = Vec::with_capacity(n * n * n);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let p = [lo + i as f64 * h, lo + j as f64 * h, lo + k as f64 * h];
                    let r2: f64 = (0..3).map(|a| (p[a] - centre[a]).powi(2)).sum();
                    rho.push((-r2 / 2.0).exp() as f32);
                }
            }
        }
        Display::Cpu { cell: sim::Cell { n, h }, origin: [lo; 3], rho: Arc::new(rho), max: 1.0 }
    }

    fn frame(p: &mut Painter, hl: &Headless, ui: &UiState, mix: f32) -> Vec<[u8; 4]> {
        p.set_view(&hl.ctx, ui, 1.0, mix);
        p.renderer.update_camera(&hl.ctx, &OrbitCamera::new(glam::Vec3::ZERO, 12.0), hl.width as f32, hl.height as f32);
        let mut enc = hl.ctx.device.create_command_encoder(&Default::default());
        p.renderer.render(&mut enc, &hl.colour_view, &hl.depth, wgpu::Color::BLACK);
        hl.ctx.queue.submit([enc.finish()]);
        hl.read_rgba()
    }

    #[test]
    fn the_risk_tint_starts_at_half_the_bound_and_is_capped() {
        assert_eq!(risk_tint(0.0), 0.0);
        assert_eq!(risk_tint(0.5), 0.0);
        assert!((risk_tint(1.25) - 0.425).abs() < 1e-6);
        assert_eq!(risk_tint(2.0), 0.85);
        assert_eq!(risk_tint(50.0), 0.85);
        let n = [SceneNucleus { z: 1, a: None, pos: [0.0; 3], vel: [0.0; 3] }];
        assert_eq!(instances(&n, None, &[])[0].colour, elements::colour(1), "no risk: no tint");
        let red = instances(&n, None, &[10.0])[0].colour;
        assert!(red[0] > red[1] && red[0] > red[2], "{red:?}");
    }

    /// The empty scene: after `clear_density` the frame is the one of a painter that never had a
    /// density (no cloud, no NaN from a stale box), and the next density snaps in.
    #[test]
    fn clearing_the_density_draws_no_cloud_and_the_next_one_snaps() {
        let Some(hl) = Headless::new(64, 64) else { return };
        let ui = UiState::new("empty", &crate::scene::Scene::preset("empty").unwrap());
        let mut fresh = Painter::new(&hl.ctx, HEADLESS_FORMAT);
        let none = frame(&mut fresh, &hl, &ui, 1.0);
        let mut p = Painter::new(&hl.ctx, HEADLESS_FORMAT);
        let a = gaussian(33, [0.0; 3]);
        assert_eq!(p.show_display(&hl.ctx, &a, &hl.depth), Ok(Shown::Snapped));
        assert_ne!(frame(&mut p, &hl, &ui, 1.0), none, "test premise: the cloud is visible");
        p.clear_density();
        assert_eq!(frame(&mut p, &hl, &ui, 1.0), none, "no cloud after clearing");
        p.rebind(&hl.ctx, &hl.depth); // a resize while empty binds nothing
        assert_eq!(frame(&mut p, &hl, &ui, 1.0), none);
        assert_eq!(p.show_display(&hl.ctx, &a, &hl.depth), Ok(Shown::Snapped), "nothing to fade from");
    }

    /// The crossfade's bookkeeping on the CPU path: the first density snaps, the next one on the
    /// same grid blends from it — uploaded into the other texture, so mix 0 still draws the first
    /// one exactly — the same density again changes nothing, and a new grid snaps (mix is then
    /// ignored: there is nothing to fade from).
    #[test]
    fn cpu_densities_ping_pong_and_snap_on_a_new_grid() {
        let Some(hl) = Headless::new(64, 64) else { return };
        let ui = UiState::new("h2", &crate::scene::Scene::preset("h2").unwrap());
        let mut p = Painter::new(&hl.ctx, HEADLESS_FORMAT);
        let (a, b) = (gaussian(33, [-1.0, 0.0, 0.0]), gaussian(33, [1.0, 0.2, 0.0]));
        assert_eq!(p.show_display(&hl.ctx, &a, &hl.depth), Ok(Shown::Snapped));
        let only_a = frame(&mut p, &hl, &ui, 0.0);
        assert_eq!(frame(&mut p, &hl, &ui, 0.0), only_a, "no previous: mix is ignored");
        assert_eq!(p.show_display(&hl.ctx, &b, &hl.depth), Ok(Shown::Blended));
        let only_b = frame(&mut p, &hl, &ui, 1.0);
        assert_ne!(only_a, only_b, "vacuous");
        assert_eq!(frame(&mut p, &hl, &ui, 0.0), only_a, "mix 0: the previous density, intact");
        assert_eq!(p.show_display(&hl.ctx, &b, &hl.depth), Ok(Shown::Same));
        assert_eq!(frame(&mut p, &hl, &ui, 0.0), only_a, "a repeated density keeps the pair");
        // a third density overwrites the first one's texture, never the one now bound as previous
        let c = gaussian(33, [0.0, 1.0, 0.0]);
        assert_eq!(p.show_display(&hl.ctx, &c, &hl.depth), Ok(Shown::Blended));
        assert_eq!(frame(&mut p, &hl, &ui, 0.0), only_b, "mix 0: b");
        // a new grid snaps
        assert_eq!(p.show_display(&hl.ctx, &gaussian(41, [0.0; 3]), &hl.depth), Ok(Shown::Snapped));
        let snapped = frame(&mut p, &hl, &ui, 1.0);
        assert_eq!(frame(&mut p, &hl, &ui, 0.0), snapped, "after a snap mix is ignored");
    }
}
