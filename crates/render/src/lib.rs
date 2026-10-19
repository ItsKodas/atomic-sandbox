pub mod camera;
pub mod dots;
pub mod elements;
pub mod glow;
pub mod headless;
pub mod lines;
pub mod spheres;
pub mod volume;

use camera::{OrbitCamera, Viewport};
use lines::LinePass;
use gpu::{GpuContext, UniformBuffer};
use spheres::{CameraUniform, SpherePass};
use volume::{ColourVolumePass, VolumePass, VolumeUniform};

/// Composes the sphere pass and the volume raymarch over one shared camera uniform. The field pass
/// that draws is one of: the scalar raymarch (`volume`; modes 0–3), the colour volume (`colour`,
/// when `draw_colour`), or the probability dots (`volume_params.mode == MODE_DOTS`). They share
/// `volume_params` ([`Renderer::sync_volume_params`] uploads it to both raymarches).
pub struct Renderer {
    pub camera_uniform: UniformBuffer<CameraUniform>, pub spheres: SpherePass, pub lines: LinePass,
    /// Translucent copies of the atoms along their recent path (the viewer's motion blur).
    pub ghosts: SpherePass,
    /// The Camera view's atoms (`glow`): drawn instead of the spheres and ghosts when `glow_on`.
    pub glow: glow::GlowPass,
    pub glow_on: bool,
    pub volume: VolumePass, pub colour: ColourVolumePass, pub volume_params: VolumeUniform, pub dots: dots::DotsPass,
    /// Draw the colour volume bound in `colour` instead of the scalar field in `volume`.
    pub draw_colour: bool,
    /// Draw the electron cloud at all (the density / orbitals / dots); false = View ▸ show ▸ None.
    pub draw_cloud: bool,
}

impl Renderer {
    /// The volume draws nothing until `volume.bind` is called and `volume_params` describes a non-empty box.
    pub fn new(ctx: &GpuContext, colour_format: wgpu::TextureFormat) -> Renderer {
        let camera_uniform = UniformBuffer::<CameraUniform>::new(ctx, "camera");
        let spheres = SpherePass::new(ctx, colour_format, &camera_uniform);
        let lines = LinePass::new(ctx, colour_format, &camera_uniform);
        let ghosts = SpherePass::with_blend(ctx, colour_format, &camera_uniform, true);
        let glow = glow::GlowPass::new(ctx, colour_format, &camera_uniform);
        let volume = VolumePass::new(ctx, colour_format, &camera_uniform);
        let colour = ColourVolumePass::new(ctx, colour_format, &camera_uniform);
        let volume_params = VolumeUniform::default();
        volume.set_params(ctx, &volume_params);
        colour.set_params(ctx, &volume_params);
        let dots = dots::DotsPass::new(ctx, colour_format, &camera_uniform);
        Renderer { camera_uniform, spheres, lines, ghosts, glow, glow_on: false, volume, colour, volume_params, dots, draw_colour: false, draw_cloud: true }
    }

    /// The camera for a scene filling the whole `width × height` target.
    pub fn update_camera(&self, ctx: &GpuContext, cam: &OrbitCamera, width: f32, height: f32) {
        self.update_camera_in(ctx, cam, Viewport::full(width, height));
    }

    /// The camera for a scene drawn into `vp` (aspect and pixel → NDC from its rect); pair with
    /// [`Renderer::render_in`] and the same `vp`.
    pub fn update_camera_in(&self, ctx: &GpuContext, cam: &OrbitCamera, vp: Viewport) {
        self.camera_uniform.write(ctx, &CameraUniform::in_viewport(cam, vp));
    }

    /// Uploads `volume_params` (call after changing it).
    pub fn sync_volume_params(&self, ctx: &GpuContext) {
        self.volume.set_params(ctx, &self.volume_params);
        self.colour.set_params(ctx, &self.volume_params);
    }

    /// One frame over the whole target: clear, spheres (depth write), lines (depth test only), then volume
    /// (reads depth). `scene_depth` is the same view bound in `volume.bind`.
    pub fn render(&self, enc: &mut wgpu::CommandEncoder, colour: &wgpu::TextureView, depth: &wgpu::TextureView, clear: wgpu::Color) {
        self.render_in(enc, colour, depth, clear, None);
    }

    /// As [`Renderer::render`], with the scene confined to `vp` inside a target of size `target`
    /// (render-pass viewport and scissor): the whole target is cleared, nothing is drawn outside `vp`.
    /// `None` draws over the whole target.
    pub fn render_in(&self, enc: &mut wgpu::CommandEncoder, colour: &wgpu::TextureView, depth: &wgpu::TextureView, clear: wgpu::Color, vp: Option<(Viewport, [u32; 2])>) {
        let confine = |pass: &mut wgpu::RenderPass<'_>| -> bool {
            let Some((v, target)) = vp else { return true };
            let Some([x, y, w, h]) = v.scissor(target) else { return false };
            pass.set_viewport(v.origin[0], v.origin[1], v.size[0], v.size[1], 0.0, 1.0);
            pass.set_scissor_rect(x, y, w, h);
            true
        };
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("spheres"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: colour, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store }, depth_slice: None })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: depth, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
            });
            if !confine(&mut pass) {
                return;
            }
            if !self.glow_on {
                self.spheres.draw(&mut pass);
            }
            self.lines.draw(&mut pass);
            if !self.glow_on {
                self.ghosts.draw(&mut pass);
            }
        }
        // `MODE_DOTS` = probability dots (`dots`) instead of the raymarch
        if !self.draw_cloud {
            if self.glow_on {
                self.glow.render(enc, colour, &confine);
            }
            return;
        }
        let dots = self.volume_params.mode == volume::MODE_DOTS;
        if dots {
            self.dots.generate(enc);
        }
        // second pass: the depth texture is sampled, so it is not attached here
        let mut pass = volume::overlay_pass(enc, colour);
        if confine(&mut pass) {
            if dots {
                self.dots.draw(&mut pass)
            } else if self.draw_colour {
                self.colour.draw(&mut pass)
            } else {
                self.volume.draw(&mut pass)
            }
        }
        drop(pass);
        if self.glow_on {
            self.glow.render(enc, colour, &confine);
        }
    }
}
