//! GPU side of the viewer, independent of where the frame goes (window surface or the headless
//! screenshot target): density volume upload, volume look, sphere instances, and the egui pass.
use crate::scene::SceneNucleus;
use crate::sim::Snapshot;
use crate::ui::UiState;
use gpu::{GpuContext, Volume3D};
use render::camera::OrbitCamera;
use render::elements;
use render::spheres::SphereInstance;
use render::volume::VolumeUniform;
use render::Renderer;

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
}

pub struct Painter {
    pub renderer: Renderer,
    volume: Option<Volume3D>,
    volume_dims: [u32; 3],
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
            volume: None,
            volume_dims: [0; 3],
            egui: egui_wgpu::Renderer::new(&ctx.device, format, options),
        }
    }

    /// Uploads a snapshot's density, recreating (and re-binding) the volume texture first if the
    /// grid changed, then auto-scales the transfer function to the density maximum. A density whose
    /// length does not match its grid is refused rather than uploaded.
    pub fn upload_snapshot(
        &mut self,
        ctx: &GpuContext,
        snap: &Snapshot,
        depth: &wgpu::TextureView,
    ) -> Result<(), String> {
        let n = snap.grid_n as u32;
        if snap.density.len() != snap.grid_n.pow(3) {
            return Err(format!(
                "snapshot density has {} values for a {}Â³ grid",
                snap.density.len(),
                snap.grid_n
            ));
        }
        if self.volume.is_none() || self.volume_dims != [n; 3] {
            let vol = Volume3D::new(ctx, [n; 3]);
            self.renderer.volume.bind(ctx, &vol, depth);
            self.volume = Some(vol);
            self.volume_dims = [n; 3];
        }
        let vol = self.volume.as_ref().expect("volume created above");
        vol.upload(ctx, &snap.density);
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
        if let Some(v) = &self.volume {
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
