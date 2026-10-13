//! GPU side of the viewer, independent of where the frame goes (window surface or the headless
//! screenshot target): density volume (upload or bind), volume look, sphere instances, and the
//! egui pass.
use crate::scene::SceneNucleus;
use crate::sim::{Density, Snapshot};
use crate::ui::UiState;
use gpu::{GpuContext, Volume3D};
use render::camera::OrbitCamera;
use render::elements;
use render::lines::LineVertex;
use render::spheres::SphereInstance;
use render::volume::VolumeUniform;
use render::Renderer;
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
}

pub struct Painter {
    pub renderer: Renderer,
    /// Upload target for CPU-backend snapshots: created by the painter, never handed out (so
    /// uploading into it cannot touch a texture the GPU worker owns).
    upload: Option<Arc<Volume3D>>,
    /// The texture the volume pass is bound to: `upload`, or the texture of a GPU-backend
    /// snapshot. Holding it keeps the GPU worker from reusing it (`sim::TexturePool`).
    shown: Option<Arc<Volume3D>>,
    egui: egui_wgpu::Renderer,
}

impl Painter {
    pub fn new(ctx: &GpuContext, format: wgpu::TextureFormat) -> Painter {
        let options = egui_wgpu::RendererOptions {
            msaa_samples: 1,
            ..Default::default()
        };
        Painter {
            renderer: Renderer::new(ctx, format),
            upload: None,
            shown: None,
            egui: egui_wgpu::Renderer::new(&ctx.device, format, options),
        }
    }

    /// Shows a snapshot's density: a CPU snapshot is uploaded into the painter's own texture
    /// (recreated on a grid change), a GPU snapshot's texture is bound as is — no copy. The
    /// volume pass is re-bound whenever the texture object changes, then the transfer function
    /// is auto-scaled to the density maximum. A density whose size does not match its grid is
    /// refused rather than shown.
    pub fn show_snapshot(
        &mut self,
        ctx: &GpuContext,
        snap: &Snapshot,
        depth: &wgpu::TextureView,
    ) -> Result<(), String> {
        let n = snap.grid_n as u32;
        if snap.density.len() != snap.grid_n.pow(3) {
            return Err(format!(
                "snapshot density has {} values for a {}³ grid",
                snap.density.len(),
                snap.grid_n
            ));
        }
        let vol = match &snap.density {
            Density::Cpu(values) => {
                if self.upload.as_ref().is_none_or(|v| v.dims != [n; 3]) {
                    self.upload = Some(Arc::new(Volume3D::new(ctx, [n; 3])));
                }
                let vol = self.upload.clone().expect("created above");
                vol.upload(ctx, values);
                vol
            }
            Density::Gpu(vol) if vol.dims == [n; 3] => vol.clone(),
            Density::Gpu(vol) => {
                return Err(format!(
                    "snapshot texture is {:?} for a {}³ grid",
                    vol.dims, snap.grid_n
                ))
            }
        };
        if self.shown.as_ref().is_none_or(|s| !Arc::ptr_eq(s, &vol)) {
            self.renderer.volume.bind(ctx, &vol, depth);
        }
        self.shown = Some(vol);
        let h = snap.grid_h as f32;
        let lo = snap.origin.map(|x| x as f32);
        let p = &mut self.renderer.volume_params;
        p.scale = snap.density_max.max(1e-12);
        p.box_min = lo;
        p.box_max = lo.map(|x| x + (n - 1) as f32 * h);
        p.step = (h / 2.0).min(0.1);
        self.renderer.sync_volume_params(ctx);
        Ok(())
    }

    /// After a resize the depth view is new: the volume pass must sample the new one.
    pub fn rebind(&mut self, ctx: &GpuContext, depth: &wgpu::TextureView) {
        if let Some(v) = &self.shown {
            self.renderer.volume.bind(ctx, v, depth);
        }
    }

    /// Copies the panel's look settings into the volume uniform; uploads only when they changed.
    pub fn set_view(&mut self, ctx: &GpuContext, ui: &UiState) {
        let old = self.renderer.volume_params;
        let new = VolumeUniform {
            mode: ui.volume_mode,
            gamma: ui.gamma,
            opacity: ui.opacity,
            iso: ui.iso,
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
        self.renderer
            .update_camera(ctx, world.camera, w as f32, h as f32);
        self.renderer
            .spheres
            .set_instances(ctx, &instances(world.nuclei, world.selected));
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
            .render(&mut enc, target.colour, target.depth, CLEAR);
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

fn instances(nuclei: &[SceneNucleus], selected: Option<usize>) -> Vec<SphereInstance> {
    nuclei
        .iter()
        .enumerate()
        .map(|(i, n)| SphereInstance {
            centre: n.pos.map(|x| x as f32),
            radius: elements::radius(n.z),
            colour: elements::colour(n.z),
            selected: if selected == Some(i) { 1.0 } else { 0.0 },
        })
        .collect()
}
