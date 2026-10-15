//! GPU side of the viewer, independent of where the frame goes (window surface or the headless
//! screenshot target): density volume (upload or bind), volume look, sphere instances, and the
//! egui pass.
//!
//! # Two layers
//! The 3D view draws one **layer** per frame: the density (`show_display`, today's view) or the
//! **orbital layer** (`show_orbitals`: one orbital ψ̃ or ψ̃², the spin density, or the colour
//! volume of all the included orbitals — orbital-viewer spec §3, §6), whichever
//! [`Painter::drawn`] says for the panel's `show` mode. Both layers keep their textures and their
//! crossfade pair (previous, current) alive, so switching the mode back and forth rebinds and
//! uploads nothing; the passes are (re)bound lazily in [`Painter::set_view`]. A layer of the
//! wrong kind for the mode (the payload of the previous mode, until the new one arrives) draws
//! nothing rather than the wrong thing.
use crate::scene::SceneNucleus;
use crate::sim::Snapshot;
use crate::ui::{ShowMode, UiState, VOLUME_DOTS, VOLUME_ISO};
use gpu::{ColourVolume3D, GpuContext, Volume3D};
use render::camera::{OrbitCamera, Viewport};
use render::elements;
use render::lines::LineVertex;
use render::spheres::SphereInstance;
use render::volume::{VolumeUniform, MODE_DOTS, MODE_ISO, MODE_SIGNED_ISO, MODE_SIGNED_VOLUME, MODE_VOLUME, PHASE_NEG, PHASE_POS, SPIN_DOWN, SPIN_UP};
use render::Renderer;
use sim::{Display, OrbitalDisplay, OrbitalField, OrbitalRequest};
use std::sync::Arc;

/// The volume mode's soft knee for a single orbital or spin product (a fraction of its own
/// maximum; transparent below an eighth of it): the low tail would otherwise swamp the lobes.
const ORBITAL_VOLUME_KNEE: f32 = 0.08;
/// Probability dots at the default 1400 × 900 view (scaled with the viewport area).
const DOTS_PER_FRAME: u32 = 200_000;
/// The spheres' size factor while an orbital layer is drawn: at full size they hide the orbital.
const ORBITAL_NUCLEUS_SCALE: f32 = 0.3;

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

/// The grid of a layer: n, h and the position of its first point (a change snaps the crossfade).
type GridKey = (usize, f64, [f64; 3]);

/// The volume pass's box and auto-scale for one layer.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Geometry {
    box_min: [f32; 3],
    box_max: [f32; 3],
    /// What 1.0 means (the density's maximum; an orbital product's scale).
    scale: f32,
    step: f32,
}

impl Geometry {
    const EMPTY: Geometry = Geometry { box_min: [0.0; 3], box_max: [0.0; 3], scale: 1.0, step: 0.1 };

    fn of(cell: &sim::Cell, origin: &[f64; 3], scale: f32) -> Geometry {
        let (n, h) = (cell.n as u32, cell.h as f32);
        let lo = origin.map(|x| x as f32);
        Geometry { box_min: lo, box_max: lo.map(|x| x + (n - 1) as f32 * h), scale: scale.max(1e-12), step: (h / 2.0).min(0.1) }
    }
}

/// An orbital layer's fields: the crossfade's two ends of a scalar product (r32float) or of the
/// colour volume (rgba16float).
enum LayerField {
    Scalar { cur: Arc<Volume3D>, prev: Option<Arc<Volume3D>> },
    Colour { cur: Arc<ColourVolume3D>, prev: Option<Arc<ColourVolume3D>> },
}

/// The orbital layer: the newest `OrbitalDisplay` field and its crossfade partner. Payloads of the
/// same request (as resolved by the producer) and grid crossfade; a change of mode, selection or
/// grid snaps (spec §7).
struct OrbitalLayer {
    what: OrbitalRequest,
    grid: GridKey,
    geometry: Geometry,
    /// ψ̃ or m (else ψ̃², or the colour volume).
    signed: bool,
    field: LayerField,
}

impl OrbitalLayer {
    /// The `show` mode this layer's payload belongs to.
    fn mode(&self) -> ShowMode {
        match self.what {
            OrbitalRequest::Orbital { .. } => ShowMode::Orbitals,
            OrbitalRequest::AllColoured { .. } => ShowMode::Coloured,
            OrbitalRequest::SpinDensity => ShowMode::Spin,
            OrbitalRequest::Off | OrbitalRequest::List => ShowMode::Density,
        }
    }

    fn has_previous(&self) -> bool {
        match &self.field {
            LayerField::Scalar { prev, .. } => prev.is_some(),
            LayerField::Colour { prev, .. } => prev.is_some(),
        }
    }
}

/// Which layer a frame draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Drawn {
    Density,
    Orbital,
    /// Nothing: an orbital mode with no product of its kind yet (or an empty scene).
    Nothing,
}

/// What the dots of this frame look like (set by `set_view`, used by `paint`).
#[derive(Clone, Copy)]
struct DotsLook {
    colour_by: u32,
    signed: bool,
    tint: [f32; 4],
    tint_neg: [f32; 4],
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
    shown_grid: Option<GridKey>,
    /// The last real (solved) density shown and its grid: the other end of the atomic
    /// superposition's blend ([`Painter::show_superposition`]).
    real: Option<(Arc<Volume3D>, GridKey)>,
    density_geometry: Geometry,
    egui: egui_wgpu::Renderer,
    /// The orbital layer ([`Painter::show_orbitals`]), its geometry, and the upload targets of
    /// CPU payloads (ping-pong, like the density's) with the CPU values last uploaded.
    orbital: Option<OrbitalLayer>,
    orbital_uploads: [Option<Arc<Volume3D>>; 2],
    orbital_colour_uploads: [Option<Arc<ColourVolume3D>>; 2],
    orbital_uploaded: Option<Arc<Vec<f32>>>,
    orbital_colour_uploaded: Option<Arc<Vec<[f32; 4]>>>,
    /// What the passes are bound to (`None`: dirty, rebind at the next [`Painter::set_view`]),
    /// and the depth view they were bound with.
    bound: Option<Drawn>,
    depth: Option<wgpu::TextureView>,
    /// The `show` mode of the last [`Painter::set_view`] (`Density` before the first): new fields
    /// are bound for it at once, so the density of a viewer that never shows orbitals is bound
    /// the moment it arrives, as before the orbital layer existed.
    mode: ShowMode,
    /// A 1³ stand-in for the scalar slot of the dots pass when it samples the colour volume.
    dummy: Volume3D,
    /// The spheres' size factor of this frame (smaller while an orbital layer is drawn).
    nucleus_scale: f32,
    /// This frame's dots: the seed (advanced every frame while shimmering) and look.
    dots_seed: u32,
    shimmer: bool,
    dots: DotsLook,
}

/// What [`Painter::show_display`] and [`Painter::show_orbitals`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shown {
    /// The field is the one already shown.
    Same,
    /// A new field, crossfaded from the previous one (same kind and grid).
    Blended,
    /// A new field with nothing to fade from (the first, or a new kind, selection or grid): snapped.
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
            density_geometry: Geometry::EMPTY,
            egui: egui_wgpu::Renderer::new(&ctx.device, format, options),
            orbital: None,
            orbital_uploads: [None, None],
            orbital_colour_uploads: [None, None],
            orbital_uploaded: None,
            orbital_colour_uploaded: None,
            bound: None,
            depth: None,
            mode: ShowMode::Density,
            dummy: Volume3D::new(ctx, [1; 3]),
            nucleus_scale: 1.0,
            dots_seed: 1,
            shimmer: true,
            dots: DotsLook { colour_by: render::dots::COLOUR_BY_DENSITY, signed: false, tint: [1.0; 4], tint_neg: [1.0; 4] },
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
    /// skipped when it is the density already uploaded), a GPU one's texture is held as is — no
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
                        let v = upload_free(ctx, &mut self.uploads, rho, n, keep.as_ref());
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
        self.depth = Some(depth.clone());
        let grid = (cell.n, cell.h, *origin);
        let shown = if self.shown.as_ref().is_some_and(|s| Arc::ptr_eq(s, &vol)) {
            Shown::Same
        } else {
            let same_grid = self.shown_grid == Some(grid);
            self.previous = if same_grid { self.shown.take() } else { None };
            self.bound = None;
            if same_grid { Shown::Blended } else { Shown::Snapped }
        };
        self.real = Some((vol.clone(), grid));
        self.shown = Some(vol);
        self.shown_grid = Some(grid);
        self.density_geometry = Geometry::of(cell, origin, max);
        self.refresh(ctx);
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
        let vol = upload_free(ctx, &mut self.uploads, rho, n, real.as_ref().map(|(r, _)| r));
        self.uploaded = None;
        let grid = (cell.n, cell.h, *origin);
        self.depth = Some(depth.clone());
        self.previous = real.filter(|(_, g)| *g == grid).map(|(v, _)| v);
        self.shown = Some(vol);
        self.shown_grid = Some(grid);
        self.density_geometry = Geometry::of(cell, origin, *max);
        self.bound = None;
        self.refresh(ctx);
        Ok(())
    }

    /// No density at all (the empty scene, `sim::dynamics` "The empty scene": no electrons): the
    /// volume pass draws nothing until the next density, which snaps in (nothing to fade from).
    /// Dropping the textures also returns the GPU provider's to its pool. The orbital layer goes
    /// too ([`Painter::clear_orbitals`]).
    pub fn clear_density(&mut self) {
        self.shown = None;
        self.previous = None;
        self.shown_grid = None;
        self.real = None;
        self.uploaded = None;
        self.clear_orbitals();
        self.unbound();
    }

    /// Back to the last real density alone (the superposition switched off).
    pub fn show_real(&mut self, ctx: &GpuContext, depth: &wgpu::TextureView) {
        if let Some((v, grid)) = self.real.clone() {
            self.depth = Some(depth.clone());
            self.previous = None;
            self.shown = Some(v);
            self.shown_grid = Some(grid);
            self.bound = None;
            self.refresh(ctx);
        }
    }

    /// Shows an orbital payload's field as the orbital layer (spec §4.2, §6): a GPU product's
    /// texture is held as is, a CPU one's values are uploaded into the painter's own textures (the
    /// one not shown; skipped when they are the values already uploaded). A payload of the same
    /// request (as resolved by the producer: another orbital, mode or inclusion snaps) on the same
    /// grid is crossfaded from the previous one — signed fields too, because the sign rule keeps
    /// consecutive fields in phase. A payload without a field (the list alone, nothing to draw)
    /// drops the layer. A field whose size does not match its grid is refused rather than shown.
    pub fn show_orbitals(&mut self, ctx: &GpuContext, o: &OrbitalDisplay, depth: &wgpu::TextureView) -> Result<Shown, String> {
        self.depth = Some(depth.clone());
        let Some(field) = o.field.as_ref() else {
            self.clear_orbitals();
            return Ok(Shown::Snapped);
        };
        let _ = ctx;
        let n = o.cell.n as u32;
        let grid = (o.cell.n, o.cell.h, o.origin);
        let same = |old: &OrbitalLayer| old.what == o.what && old.grid == grid;
        let check = |dims: [u32; 3]| -> Result<(), String> {
            if dims == [n; 3] { Ok(()) } else { Err(format!("orbital texture is {dims:?} for a {}³ grid", o.cell.n)) }
        };
        let (new, scale, signed) = match field {
            OrbitalField::Gpu { volume, signed, scale } => {
                check(volume.dims)?;
                self.orbital_uploaded = None;
                (LayerField::Scalar { cur: volume.clone(), prev: None }, *scale, *signed)
            }
            OrbitalField::Cpu { values, signed, scale } => {
                if values.len() != o.cell.n.pow(3) {
                    return Err(format!("orbital field has {} values for a {}³ grid", values.len(), o.cell.n));
                }
                let current = match self.orbital.as_ref().map(|l| &l.field) {
                    Some(LayerField::Scalar { cur, .. }) => Some(cur.clone()),
                    _ => None,
                };
                let vol = match current {
                    Some(v) if self.orbital_uploaded.as_ref().is_some_and(|u| Arc::ptr_eq(u, values)) => v,
                    keep => {
                        let v = upload_free(ctx, &mut self.orbital_uploads, values, n, keep.as_ref());
                        self.orbital_uploaded = Some(values.clone());
                        v
                    }
                };
                (LayerField::Scalar { cur: vol, prev: None }, *scale, *signed)
            }
            OrbitalField::GpuColour { volume, scale } => {
                check(volume.dims)?;
                self.orbital_colour_uploaded = None;
                (LayerField::Colour { cur: volume.clone(), prev: None }, *scale, false)
            }
            OrbitalField::CpuColour { rgba, scale } => {
                if rgba.len() != o.cell.n.pow(3) {
                    return Err(format!("orbital colour field has {} values for a {}³ grid", rgba.len(), o.cell.n));
                }
                let current = match self.orbital.as_ref().map(|l| &l.field) {
                    Some(LayerField::Colour { cur, .. }) => Some(cur.clone()),
                    _ => None,
                };
                let vol = match current {
                    Some(v) if self.orbital_colour_uploaded.as_ref().is_some_and(|u| Arc::ptr_eq(u, rgba)) => v,
                    keep => {
                        let v = upload_free_colour(ctx, &mut self.orbital_colour_uploads, rgba, n, keep.as_ref());
                        self.orbital_colour_uploaded = Some(rgba.clone());
                        v
                    }
                };
                (LayerField::Colour { cur: vol, prev: None }, *scale, false)
            }
        };
        let geometry = Geometry::of(&o.cell, &o.origin, scale);
        // the crossfade partner: the old current field, when this is the same request on the same
        // grid and a different texture (the same one is a repeated payload)
        let (field, shown) = match (self.orbital.take(), new) {
            (Some(old), LayerField::Scalar { cur, .. }) if same(&old) => match old.field {
                LayerField::Scalar { cur: oc, prev } if Arc::ptr_eq(&oc, &cur) => (LayerField::Scalar { cur, prev }, Shown::Same),
                LayerField::Scalar { cur: oc, .. } => (LayerField::Scalar { cur, prev: Some(oc) }, Shown::Blended),
                LayerField::Colour { .. } => (LayerField::Scalar { cur, prev: None }, Shown::Snapped),
            },
            (Some(old), LayerField::Colour { cur, .. }) if same(&old) => match old.field {
                LayerField::Colour { cur: oc, prev } if Arc::ptr_eq(&oc, &cur) => (LayerField::Colour { cur, prev }, Shown::Same),
                LayerField::Colour { cur: oc, .. } => (LayerField::Colour { cur, prev: Some(oc) }, Shown::Blended),
                LayerField::Scalar { .. } => (LayerField::Colour { cur, prev: None }, Shown::Snapped),
            },
            (_, new) => (new, Shown::Snapped),
        };
        self.orbital = Some(OrbitalLayer { what: o.what.clone(), grid, geometry, signed, field });
        if shown != Shown::Same {
            self.bound = None;
        }
        self.refresh(ctx);
        Ok(shown)
    }

    /// Drops the orbital layer (the view switched off, an empty scene, a payload with nothing to
    /// draw): the textures return to the provider's pool.
    pub fn clear_orbitals(&mut self) {
        if self.orbital.take().is_some() {
            self.unbound();
        }
        self.orbital_uploaded = None;
        self.orbital_colour_uploaded = None;
    }

    /// A field went away: the passes are dirty, and if the mode now has nothing to draw they let go
    /// of their textures at once (no context needed to bind nothing).
    fn unbound(&mut self) {
        self.bound = None;
        if self.drawn_for(self.mode) == Drawn::Nothing {
            let r = &mut self.renderer;
            (r.volume.unbind(), r.colour.unbind(), r.dots.unbind());
            r.draw_colour = false;
            self.bound = Some(Drawn::Nothing);
        }
    }

    /// Brings the passes to what the current mode draws, when that changed.
    fn refresh(&mut self, ctx: &GpuContext) {
        let want = self.drawn_for(self.mode);
        if self.bound != Some(want) {
            self.bind(ctx, want);
        }
    }

    /// Which layer a frame draws for the panel's settings (module doc, "Two layers").
    pub fn drawn(&self, ui: &UiState) -> Drawn {
        self.drawn_for(ui.show)
    }

    fn drawn_for(&self, mode: ShowMode) -> Drawn {
        match mode {
            ShowMode::Density if self.shown.is_some() => Drawn::Density,
            ShowMode::Density => Drawn::Nothing,
            mode => match &self.orbital {
                Some(l) if l.mode() == mode => Drawn::Orbital,
                _ => Drawn::Nothing,
            },
        }
    }

    /// Binds the passes to what `drawn` says: the density pair, the orbital scalar pair, the
    /// orbital colour pair, or nothing (the dots pass samples whichever field is drawn).
    fn bind(&mut self, ctx: &GpuContext, drawn: Drawn) {
        let Some(depth) = self.depth.clone() else { return };
        let r = &mut self.renderer;
        r.volume.unbind();
        r.colour.unbind();
        r.dots.unbind();
        r.draw_colour = false;
        match (drawn, &self.orbital, &self.shown) {
            (Drawn::Density, _, Some(cur)) => {
                match &self.previous {
                    Some(prev) => r.volume.bind_pair(ctx, prev, cur, &depth),
                    None => r.volume.bind(ctx, cur, &depth),
                }
                r.dots.bind(ctx, cur, None, &depth);
            }
            (Drawn::Orbital, Some(layer), _) => match &layer.field {
                LayerField::Scalar { cur, prev } => {
                    r.volume.bind_pair(ctx, prev.as_ref().unwrap_or(cur), cur, &depth);
                    r.dots.bind(ctx, cur, None, &depth);
                }
                LayerField::Colour { cur, prev } => {
                    r.colour.bind_pair(ctx, prev.as_ref().unwrap_or(cur), cur, &depth);
                    r.dots.bind(ctx, &self.dummy, Some(cur), &depth);
                    r.draw_colour = true;
                }
            },
            _ => {}
        }
        // the box and scale of what is bound, at once (`set_view` sets the look)
        let geometry = match (drawn, &self.orbital) {
            (Drawn::Orbital, Some(l)) => l.geometry,
            _ => self.density_geometry,
        };
        let u = &mut self.renderer.volume_params;
        (u.box_min, u.box_max, u.scale, u.step) = (geometry.box_min, geometry.box_max, geometry.scale, geometry.step);
        self.renderer.sync_volume_params(ctx);
        self.bound = Some(drawn);
    }

    /// After a resize the depth view is new: the volume pass must sample the new one.
    pub fn rebind(&mut self, ctx: &GpuContext, depth: &wgpu::TextureView) {
        self.depth = Some(depth.clone());
        self.bound = None;
        self.refresh(ctx);
    }

    /// Copies the panel's look settings, the layer's `fade` (1 = current, less = the last DFT
    /// density or orbitals behind a surrogate step: `viewer::DensityShown::fade`,
    /// `viewer::OrbitalShown::fade`) and the crossfade's `mix` (0 = the previous field, 1 = the
    /// current: `viewer::Viewer::density_mix`; 1 when there is nothing to fade from) into the
    /// volume uniform; uploads only when they changed. Binds the passes first when the layer
    /// or its textures changed ([`Painter::drawn`]).
    pub fn set_view(&mut self, ctx: &GpuContext, ui: &UiState, fade: f32, mix: f32) {
        self.mode = ui.show;
        let drawn = self.drawn(ui);
        self.refresh(ctx);
        self.shimmer = ui.shimmer;
        self.nucleus_scale = ui.nucleus_scale.unwrap_or(if drawn == Drawn::Orbital { ORBITAL_NUCLEUS_SCALE } else { 1.0 });
        let dots = ui.volume_mode == VOLUME_DOTS;
        let scalar_mode = |dots: bool| if dots { MODE_DOTS } else if ui.volume_mode == VOLUME_ISO { MODE_ISO } else { MODE_VOLUME };
        let mut u = VolumeUniform {
            gamma: ui.gamma,
            opacity: ui.opacity,
            iso: ui.iso,
            fade,
            ..Default::default()
        };
        let (geometry, has_previous) = match (drawn, &self.orbital) {
            (Drawn::Orbital, Some(l)) => (l.geometry, l.has_previous()),
            _ => (self.density_geometry, self.previous.is_some()),
        };
        (u.box_min, u.box_max, u.scale, u.step) = (geometry.box_min, geometry.box_max, geometry.scale, geometry.step);
        u.mix = if has_previous { mix.clamp(0.0, 1.0) } else { 1.0 };
        self.dots = DotsLook { colour_by: render::dots::COLOUR_BY_DENSITY, signed: false, tint: [1.0; 4], tint_neg: [1.0; 4] };
        match (drawn, &self.orbital) {
            (Drawn::Orbital, Some(l)) => match &l.field {
                LayerField::Colour { .. } => {
                    u.mode = scalar_mode(dots);
                    u.additive = u32::from(ui.colour_additive);
                    if ui.dots_by_orbital {
                        self.dots.colour_by = render::dots::COLOUR_BY_ORBITAL;
                    }
                }
                LayerField::Scalar { .. } if l.signed => {
                    let spin = matches!(l.what, OrbitalRequest::SpinDensity);
                    let (pos, neg) = if spin { (SPIN_UP, SPIN_DOWN) } else { (PHASE_POS, PHASE_NEG) };
                    (u.colour_pos, u.colour_neg) = (pos, neg);
                    u.iso = if spin { ui.spin_iso } else { ui.phase_iso };
                    u.knee = ORBITAL_VOLUME_KNEE;
                    u.mode = if dots { MODE_DOTS } else if ui.volume_mode == VOLUME_ISO { MODE_SIGNED_ISO } else { MODE_SIGNED_VOLUME };
                    self.dots = DotsLook {
                        colour_by: render::dots::COLOUR_BY_SIGN,
                        signed: true,
                        tint: [pos[0], pos[1], pos[2], 1.0],
                        tint_neg: [neg[0], neg[1], neg[2], 1.0],
                    };
                }
                LayerField::Scalar { .. } => {
                    // ψ̃²: today's volume / isosurface / dots with today's ramp, the soft knee for the
                    // volume so the low tail does not swamp the lobes
                    u.mode = scalar_mode(dots);
                    u.knee = ORBITAL_VOLUME_KNEE;
                }
            },
            _ => u.mode = scalar_mode(dots),
        }
        if u != self.renderer.volume_params {
            self.renderer.volume_params = u;
            self.renderer.sync_volume_params(ctx);
        }
    }

    /// The probability dots' parameters for a frame in a viewport of `area` pixels² (dots mode
    /// only): a new seed every frame while shimmering, the count scaled with the area.
    fn update_dots(&mut self, ctx: &GpuContext, area: f32) {
        if self.renderer.volume_params.mode != MODE_DOTS {
            return;
        }
        if self.shimmer {
            self.dots_seed = self.dots_seed.wrapping_add(1);
        }
        let p = &self.renderer.volume_params;
        let count = (DOTS_PER_FRAME as f32 * area.max(1.0) / (1400.0 * 900.0)).clamp(50_000.0, render::dots::MAX_DOTS as f32) as u32;
        let look = self.dots;
        self.renderer.dots.set_params(ctx, &render::dots::DotsUniform {
            box_min: p.box_min, use_colour: 0, box_max: p.box_max, count, dims: [0; 3], seed: self.dots_seed,
            tint: if look.colour_by == render::dots::COLOUR_BY_SIGN { look.tint } else { [0.55, 0.8, 1.0, 1.0] },
            tint_neg: look.tint_neg, size_px: 1.25, intensity: 0.35 * p.fade,
            colour_by: look.colour_by, signed: u32::from(look.signed),
        });
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
            .set_instances(ctx, &instances(world.nuclei, world.selected, world.risk, self.nucleus_scale));
        self.renderer.lines.set_lines(ctx, world.lines);
        self.update_dots(ctx, vp.size[0] * vp.size[1]);
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

/// Uploads `data` into the slot of `slots` that is not `keep` (created, or recreated for new
/// dims) and returns it.
fn upload_free(ctx: &GpuContext, slots: &mut [Option<Arc<Volume3D>>; 2], data: &[f32], n: u32, keep: Option<&Arc<Volume3D>>) -> Arc<Volume3D> {
    let holds = |v: &Option<Arc<Volume3D>>| matches!((v, keep), (Some(v), Some(k)) if Arc::ptr_eq(v, k));
    let slot = if holds(&slots[0]) { 1 } else { 0 };
    if slots[slot].as_ref().is_none_or(|v| v.dims != [n; 3]) {
        slots[slot] = Some(Arc::new(Volume3D::new(ctx, [n; 3])));
    }
    let vol = slots[slot].clone().expect("created above");
    vol.upload(ctx, data);
    vol
}

/// [`upload_free`] for the colour volume (f32 RGBA rounded to f16 at upload).
fn upload_free_colour(ctx: &GpuContext, slots: &mut [Option<Arc<ColourVolume3D>>; 2], data: &[[f32; 4]], n: u32, keep: Option<&Arc<ColourVolume3D>>) -> Arc<ColourVolume3D> {
    let holds = |v: &Option<Arc<ColourVolume3D>>| matches!((v, keep), (Some(v), Some(k)) if Arc::ptr_eq(v, k));
    let slot = if holds(&slots[0]) { 1 } else { 0 };
    if slots[slot].as_ref().is_none_or(|v| v.dims != [n; 3]) {
        slots[slot] = Some(Arc::new(ColourVolume3D::new(ctx, [n; 3])));
    }
    let vol = slots[slot].clone().expect("created above");
    vol.upload_f32(ctx, data);
    vol
}

/// Draft mode's tint: the fraction of red mixed into a sphere at `risk` (the larger of its force
/// uncertainty ÷ bound and its envelope distance ÷ limit): none at or below half the bound,
/// growing to 0.85 at twice it (capped, so the element colour still shows).
pub fn risk_tint(risk: f32) -> f32 {
    ((risk - 0.5) / 1.5).clamp(0.0, 1.0) * 0.85
}

const RISK_RED: [f32; 3] = [1.0, 0.12, 0.08];

fn instances(nuclei: &[SceneNucleus], selected: Option<usize>, risk: &[f32], scale: f32) -> Vec<SphereInstance> {
    nuclei
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let base = elements::colour(n.z);
            let t = risk.get(i).map_or(0.0, |&r| risk_tint(r));
            SphereInstance {
                centre: n.pos.map(|x| x as f32),
                radius: elements::radius(n.z) * scale,
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
        p.update_dots(&hl.ctx, (hl.width * hl.height) as f32);
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
        assert_eq!(instances(&n, None, &[], 1.0)[0].colour, elements::colour(1), "no risk: no tint");
        let red = instances(&n, None, &[10.0], 1.0)[0].colour;
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

    // ---- the orbital layer ----

    use crate::ui::ShowMode;
    use sim::{LoneAtomLabels, SpinSummary};

    /// An orbital payload around `field`: the geometry of [`gaussian`]'s grid, `what` as the producer
    /// resolved it, no rows (the painter reads only the geometry, the request and the field).
    fn payload(n: usize, what: OrbitalRequest, field: Option<OrbitalField>) -> OrbitalDisplay {
        let lo = -(n as f64 - 1.0) * 0.25 / 2.0;
        OrbitalDisplay {
            cell: sim::Cell { n, h: 0.25 },
            origin: [lo; 3],
            positions: Vec::new(),
            rows: Vec::new(),
            field,
            spin: SpinSummary::default(),
            unoccupied: None,
            notes: Vec::new(),
            what,
            lone_atom: LoneAtomLabels::NotLoneAtom,
        }
    }

    /// x e^{−r²/2} (a p_x-like orbital) on an n³ grid of spacing 0.25, shifted by `dx` along x.
    fn px_values(n: usize, dx: f64) -> Arc<Vec<f32>> {
        let lo = -(n as f64 - 1.0) * 0.25 / 2.0;
        let mut v = Vec::with_capacity(n * n * n);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let (x, y, z) = (lo + i as f64 * 0.25 - dx, lo + j as f64 * 0.25, lo + k as f64 * 0.25);
                    v.push((x * (-(x * x + y * y + z * z) / 2.0).exp()) as f32);
                }
            }
        }
        Arc::new(v)
    }

    fn orbital(track: u32) -> OrbitalRequest {
        OrbitalRequest::Orbital { spin: sim::Spin::Up, track, squared: false }
    }

    /// The orbital layer's bookkeeping, as the density's: the first payload snaps, the next of the
    /// same request on the same grid blends from it — mix 0 still draws the first one exactly —
    /// a repeated one changes nothing, and another orbital or grid snaps. A layer of the wrong
    /// kind for the `show` mode draws nothing, and the density comes back with the mode.
    #[test]
    fn orbital_payloads_crossfade_per_request_and_snap_otherwise() {
        let Some(hl) = Headless::new(64, 64) else { return };
        let mut ui = UiState::new("h2", &crate::scene::Scene::preset("h2").unwrap());
        let mut p = Painter::new(&hl.ctx, HEADLESS_FORMAT);
        let density = gaussian(33, [0.0; 3]);
        p.show_display(&hl.ctx, &density, &hl.depth).unwrap();
        let only_density = frame(&mut p, &hl, &ui, 1.0);
        ui.show = ShowMode::Orbitals;
        assert_eq!(p.drawn(&ui), Drawn::Nothing, "no orbital payload yet");
        let nothing = frame(&mut p, &hl, &ui, 1.0);
        assert_ne!(nothing, only_density, "the density is not drawn in an orbital mode");
        let a = OrbitalField::Cpu { values: px_values(33, 0.0), signed: true, scale: 0.6065 };
        let b = OrbitalField::Cpu { values: px_values(33, 1.0), signed: true, scale: 0.6065 };
        assert_eq!(p.show_orbitals(&hl.ctx, &payload(33, orbital(3), Some(a.clone())), &hl.depth), Ok(Shown::Snapped));
        assert_eq!(p.drawn(&ui), Drawn::Orbital);
        let only_a = frame(&mut p, &hl, &ui, 1.0);
        assert_ne!(only_a, nothing, "test premise: the orbital is visible");
        assert_eq!(frame(&mut p, &hl, &ui, 0.0), only_a, "no previous: mix is ignored");
        assert_eq!(p.show_orbitals(&hl.ctx, &payload(33, orbital(3), Some(a.clone())), &hl.depth), Ok(Shown::Same), "the same values again");
        assert_eq!(p.show_orbitals(&hl.ctx, &payload(33, orbital(3), Some(b.clone())), &hl.depth), Ok(Shown::Blended));
        let only_b = frame(&mut p, &hl, &ui, 1.0);
        assert_ne!(only_a, only_b, "vacuous");
        assert_eq!(frame(&mut p, &hl, &ui, 0.0), only_a, "mix 0: the previous payload, intact");
        // another orbital (a different track) snaps: nothing to fade from
        assert_eq!(p.show_orbitals(&hl.ctx, &payload(33, orbital(4), Some(a.clone())), &hl.depth), Ok(Shown::Snapped));
        let snapped = frame(&mut p, &hl, &ui, 0.0);
        assert_eq!(frame(&mut p, &hl, &ui, 1.0), snapped, "after a snap mix is ignored");
        // a new grid snaps
        assert_eq!(p.show_orbitals(&hl.ctx, &payload(41, orbital(4), Some(OrbitalField::Cpu { values: px_values(41, 0.0), signed: true, scale: 0.6065 })), &hl.depth), Ok(Shown::Snapped));
        // the wrong kind for the mode draws nothing; the right kind does
        ui.show = ShowMode::Spin;
        assert_eq!(p.drawn(&ui), Drawn::Nothing, "an orbital is not the spin density");
        ui.show = ShowMode::Density;
        assert_eq!(p.drawn(&ui), Drawn::Density);
        assert_eq!(frame(&mut p, &hl, &ui, 1.0), only_density, "the density is back, untouched by the orbital layer");
        // a malformed field is refused, a payload without a field drops the layer
        assert!(p.show_orbitals(&hl.ctx, &payload(33, orbital(4), Some(OrbitalField::Cpu { values: Arc::new(vec![0.0; 5]), signed: true, scale: 1.0 })), &hl.depth).is_err());
        ui.show = ShowMode::Orbitals;
        p.show_orbitals(&hl.ctx, &payload(33, OrbitalRequest::List, None), &hl.depth).unwrap();
        assert_eq!(p.drawn(&ui), Drawn::Nothing);
    }

    /// The three kinds of orbital product are bound where they belong: the signed scalar (phase
    /// colours: red and blue lobes), the spin density (amber and teal) and the colour volume (the
    /// orbital's own slot colour, not the density ramp), from GPU textures as the providers hand
    /// them out (an `Arc` the painter holds) and from CPU values; the dots run on each.
    #[test]
    fn each_kind_of_product_draws_in_its_own_colours() {
        let Some(hl) = Headless::new(96, 96) else { return };
        let mut p = Painter::new(&hl.ctx, HEADLESS_FORMAT);
        let mut ui = UiState::new("h2", &crate::scene::Scene::preset("h2").unwrap());
        ui.volume_mode = crate::ui::VOLUME_ISO;
        let values = px_values(33, 0.0);
        let vol = Arc::new(Volume3D::new(&hl.ctx, [33; 3]));
        vol.upload(&hl.ctx, &values);
        let lit = |f: &[[u8; 4]]| f.iter().filter(|px| px[..3].iter().map(|&c| c as u32).sum::<u32>() > 40).count();
        let mean = |f: &[[u8; 4]]| {
            let (mut r, mut g, mut b, mut n) = (0u64, 0u64, 0u64, 0u64);
            for px in f.iter().filter(|px| px[..3].iter().map(|&c| c as u32).sum::<u32>() > 40) {
                (r, g, b, n) = (r + px[0] as u64, g + px[1] as u64, b + px[2] as u64, n + 1);
            }
            [r as f64 / n as f64, g as f64 / n as f64, b as f64 / n as f64]
        };
        // phase: both lobes drawn, red and blue both present
        ui.show = ShowMode::Orbitals;
        let field = OrbitalField::Gpu { volume: vol.clone(), signed: true, scale: 0.6065 };
        p.show_orbitals(&hl.ctx, &payload(33, orbital(1), Some(field.clone())), &hl.depth).unwrap();
        let phase = frame(&mut p, &hl, &ui, 1.0);
        assert!(lit(&phase) > 200, "the ± isosurface is drawn ({} lit pixels)", lit(&phase));
        let has = |f: &[[u8; 4]], want: fn([u8; 4]) -> bool| f.iter().any(|&px| want(px));
        assert!(has(&phase, |px| px[0] as i32 - px[2] as i32 > 60), "a red (+) lobe");
        assert!(has(&phase, |px| px[2] as i32 - px[0] as i32 > 60), "a blue (−) lobe");
        // the same field as the spin density: amber and teal instead
        ui.show = ShowMode::Spin;
        p.show_orbitals(&hl.ctx, &payload(33, OrbitalRequest::SpinDensity, Some(field)), &hl.depth).unwrap();
        let spin = frame(&mut p, &hl, &ui, 1.0);
        assert!(has(&spin, |px| px[0] as i32 - px[2] as i32 > 60 && px[1] > 100), "an amber (↑ excess) lobe");
        assert!(has(&spin, |px| px[1] as i32 - px[0] as i32 > 60), "a teal (↓ excess) lobe");
        // the colour volume from CPU values, both blends
        ui.show = ShowMode::Coloured;
        let rgba: Arc<Vec<[f32; 4]>> = Arc::new(values.iter().map(|&v| { let s = v * v / 0.3679; [0.9 * s, 0.4 * s, 0.1 * s, s] }).collect());
        let cf = OrbitalField::CpuColour { rgba: rgba.clone(), scale: 1.0 };
        let what = OrbitalRequest::AllColoured { include: None, physical: false, additive: true };
        assert_eq!(p.show_orbitals(&hl.ctx, &payload(33, what.clone(), Some(cf)), &hl.depth), Ok(Shown::Snapped));
        ui.volume_mode = crate::ui::VOLUME_VOLUME;
        ui.colour_additive = true;
        let additive = frame(&mut p, &hl, &ui, 1.0);
        ui.colour_additive = false;
        let weighted = frame(&mut p, &hl, &ui, 1.0);
        let (ma, mw) = (mean(&additive), mean(&weighted));
        eprintln!("colour volume: additive mean {ma:?}, weighted mean {mw:?}");
        assert!(lit(&additive) > 200 && lit(&weighted) > 200);
        assert!(ma[0] > ma[2] && mw[0] > mw[2], "the orbital's slot colour (orange) shows, not the density ramp");
        // dots on each kind run without a panic and draw something
        ui.volume_mode = crate::ui::VOLUME_DOTS;
        for (mode, w) in [(ShowMode::Coloured, what.clone()), (ShowMode::Spin, OrbitalRequest::SpinDensity)] {
            ui.show = mode;
            if mode == ShowMode::Spin {
                p.show_orbitals(&hl.ctx, &payload(33, w, Some(OrbitalField::Gpu { volume: vol.clone(), signed: true, scale: 0.6065 })), &hl.depth).unwrap();
            }
            let dots = frame(&mut p, &hl, &ui, 1.0);
            assert!(lit(&dots) > 100, "{mode:?}: the dots are drawn ({} lit pixels)", lit(&dots));
        }
    }
}
