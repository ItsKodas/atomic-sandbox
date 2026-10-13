pub mod camera;
pub mod elements;
pub mod headless;
pub mod lines;
pub mod spheres;
pub mod volume;

use camera::OrbitCamera;
use lines::LinePass;
use gpu::{GpuContext, UniformBuffer};
use spheres::{CameraUniform, SpherePass};
use volume::{VolumePass, VolumeUniform};

/// Composes the sphere pass and the volume raymarch over one shared camera uniform.
pub struct Renderer { pub camera_uniform: UniformBuffer<CameraUniform>, pub spheres: SpherePass, pub lines: LinePass, pub volume: VolumePass, pub volume_params: VolumeUniform }

impl Renderer {
    /// The volume draws nothing until `volume.bind` is called and `volume_params` describes a non-empty box.
    pub fn new(ctx: &GpuContext, colour_format: wgpu::TextureFormat) -> Renderer {
        let camera_uniform = UniformBuffer::<CameraUniform>::new(ctx, "camera");
        let spheres = SpherePass::new(ctx, colour_format, &camera_uniform);
        let lines = LinePass::new(ctx, colour_format, &camera_uniform);
        let volume = VolumePass::new(ctx, colour_format, &camera_uniform);
        let volume_params = VolumeUniform::default();
        volume.set_params(ctx, &volume_params);
        Renderer { camera_uniform, spheres, lines, volume, volume_params }
    }

    pub fn update_camera(&self, ctx: &GpuContext, cam: &OrbitCamera, width: f32, height: f32) {
        self.camera_uniform.write(ctx, &CameraUniform::from_camera(cam, width, height));
    }

    /// Uploads `volume_params` (call after changing it).
    pub fn sync_volume_params(&self, ctx: &GpuContext) { self.volume.set_params(ctx, &self.volume_params); }

    /// One frame: clear, spheres (depth write), lines (depth test only), then volume (reads depth). `scene_depth` is the same view bound in `volume.bind`.
    pub fn render(&self, enc: &mut wgpu::CommandEncoder, colour: &wgpu::TextureView, depth: &wgpu::TextureView, clear: wgpu::Color) {
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("spheres"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: colour, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store }, depth_slice: None })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: depth, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
            });
            self.spheres.draw(&mut pass);
            self.lines.draw(&mut pass);
        }
        // second pass: the depth texture is sampled, so it is not attached here
        let mut pass = volume::overlay_pass(enc, colour);
        self.volume.draw(&mut pass);
    }
}
