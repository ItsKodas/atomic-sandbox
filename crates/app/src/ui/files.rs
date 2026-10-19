//! The File window (user request 2026-10-19): a window in the middle of the app that browses the
//! scenes (the presets, compiled in, and the `.scene` files in the `scenes` folder — `scene::scenes_dir`)
//! with a small drawing of each, and the models in the `models` folder (`scene::models_dir`), each
//! marked against the selected scene: ready, loads after applying its settings (the load card,
//! `model_offer`), or cannot run here (and why). The current scene and model save into the folders
//! from here too. Every button goes through the ordinary `Action`s (`Preset`, `LoadScene`,
//! `SaveScene`, `LoadModel`, `SaveModel`), so `--record` and the worker see nothing new.
//!
//! The folders are read when the window opens and on ↻ (never per frame); the comparison with
//! the selected scene is the load card's own (`model_offer::offer`), a pure function.
use super::{model_offer, theme, Action, PanelInput, SceneSource, UiState};
use crate::scene::Scene;
use crate::sim::Backend;
use egui::{Color32, RichText, Sense, Stroke, Ui, Vec2};
use std::path::{Path, PathBuf};

/// One scene the window offers, as it would load (a preset at the panel's quality and policy).
pub struct SceneEntry {
    name: String,
    source: SceneSource,
    scene: Result<Scene, String>,
}

pub struct ModelEntry {
    name: String,
    path: PathBuf,
    prov: Result<surrogate::Provenance, String>,
}

#[derive(Default)]
pub struct Browser {
    pub open: bool,
    selected: usize,
    scenes: Vec<SceneEntry>,
    models: Vec<ModelEntry>,
    scene_name: String,
    /// The Save window (`File ▸ Save…`): open, what to save, the name, the folders or elsewhere.
    pub save_open: bool,
    save_kind: SaveKind,
    save_name: String,
    save_elsewhere: bool,
    /// The scenes area's height once the splitter was dragged (`None`: about 2.5 rows).
    scenes_h: Option<f32>,
    /// A scene file whose grid spacing or ionic potential differs from the current settings, waiting
    /// for the choice: open it as saved, or with the current settings (user request 2026-10-19).
    confirm: Option<usize>,
    /// A Load in flight: the file, the model's (samples, epoch) when it was clicked, and since when
    /// (kept fresh while the settings card is up). Cleared when the model changes, or after
    /// [`LOAD_GIVE_UP`] with nothing loaded (refused, or the card cancelled).
    loading: Option<(PathBuf, (usize, u64), std::time::Instant)>,
    /// The file the model in force came from, and its (samples, epoch) then: its Load button stays
    /// disabled until the model changes (training, another load, Forget).
    loaded: Option<(PathBuf, (usize, u64))>,
    /// A save was sent: read the folders again at this time.
    refresh_at: Option<std::time::Instant>,
}

/// Opens the window and reads the folders.
pub fn open(st: &mut UiState) {
    st.files.open = true;
    refresh(st);
}

fn refresh(st: &mut UiState) {
    let mut scenes: Vec<SceneEntry> = Scene::PRESETS
        .iter()
        .map(|name| SceneEntry {
            name: name.to_string(),
            source: SceneSource::Preset(name.to_string()),
            scene: crate::quality::preset(name, st.quality_h)
                .map(|mut s| {
                    s.ionic = st.ionic;
                    s
                })
                .ok_or_else(|| "unknown preset".to_string()),
        })
        .collect();
    for path in files_in(crate::scene::scenes_dir(), "scene") {
        let stem = stem(&path);
        if Scene::PRESETS.contains(&stem.as_str()) {
            continue; // the dist folder's copies of the presets
        }
        let p = path.display().to_string();
        scenes.push(SceneEntry {
            name: stem,
            scene: super::load(&p),
            source: SceneSource::File(p),
        });
    }
    let models = files_in(crate::scene::models_dir(), "model")
        .into_iter()
        .map(|path| ModelEntry {
            name: stem(&path),
            prov: surrogate::sidecar::read_provenance(&path).map_err(|e| e.to_string()),
            path,
        })
        .collect();
    let b = &mut st.files;
    b.selected = scenes
        .iter()
        .position(|e| e.source == st.source)
        .unwrap_or(0);
    b.scenes = scenes;
    b.models = models;
    if b.scene_name.is_empty() {
        b.scene_name = st.source.preset().unwrap_or("my").to_string();
    }
}

fn files_in(dir: Option<PathBuf>, ext: &str) -> Vec<PathBuf> {
    let Some(dir) = dir else { return Vec::new() };
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case(ext)))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

fn stem(p: &Path) -> String {
    p.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// The scene's elements against what a model was taught.
struct Species {
    /// Each element of the scene and whether the model was taught it.
    scene: Vec<(u8, bool)>,
    /// The model was taught exactly the scene's elements, no more.
    exact: bool,
}

impl Species {
    fn covers(&self) -> bool {
        self.scene.iter().all(|&(_, t)| t)
    }
}

/// The model to recommend for a scene: one that knows every element and can run here — ready
/// before needing settings, then taught exactly these elements, then the fewest settings to
/// change; the first such on a tie.
fn recommended(fits: &[(Fit, Option<Species>)]) -> Option<usize> {
    fits.iter()
        .enumerate()
        .filter_map(|(i, (f, sp))| {
            let sp = sp.as_ref().filter(|sp| sp.covers())?;
            // ready first; then taught exactly these elements; then the fewest settings to change
            let (ready, changes) = match f {
                Fit::Ready => (1, 0),
                Fit::Settings(n, _) => (0, *n),
                _ => return None,
            };
            Some((ready, sp.exact as u8, std::cmp::Reverse(changes), std::cmp::Reverse(i)))
        })
        .max()
        .map(|(_, _, _, std::cmp::Reverse(i))| i)
}

/// Where the model stands against a scene.
enum Fit {
    Ready,
    /// Loads after the card applies these settings.
    /// Loads after the card applies these settings (how many differ, and their words).
    Settings(usize, String),
    Blocked(String),
    Unreadable(String),
}

fn fit(
    st: &UiState,
    input: &PanelInput<'_>,
    m: &ModelEntry,
    scene: &Scene,
) -> (Fit, Option<Species>) {
    let prov = match &m.prov {
        Ok(p) => p,
        Err(e) => return (Fit::Unreadable(e.clone()), None),
    };
    // the elements: the model answers only the species it was taught (the rest fall back to DFT)
    let taught: Vec<u8> = prov.masses.iter().map(|&(z, _)| z).collect();
    let mut elements: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
    elements.sort_unstable();
    elements.dedup();
    let species = (!taught.is_empty()).then(|| {
        let mut t = taught.clone();
        t.sort_unstable();
        t.dedup();
        Species { scene: elements.iter().map(|&z| (z, taught.contains(&z))).collect(), exact: t == elements }
    });
    let teacher = input.model().map_or(
        if st.gpu_available {
            "Kohn-Sham (GPU)"
        } else {
            "Kohn-Sham (CPU)"
        },
        |m| m.teacher,
    );
    let vram = |s: &Scene| {
        crate::quality::vram_check(
            s,
            crate::quality::gpu_solvers(st.backend, st.gpu_available),
            crate::quality::vram_budget_bytes(),
        )
    };
    let ctx = model_offer::Context {
        teacher,
        sample_tol: crate::worker::SAMPLE_TOL,
        pseudised: !super::pseudo::pseudised(input).is_empty(),
        vram: &vram,
    };
    let f = match model_offer::offer(&m.path, prov, scene, &ctx) {
        None => Fit::Ready,
        Some(o) => match o.blocked() {
            Some(why) => Fit::Blocked(why.to_string()),
            None => Fit::Settings(
                o.rows.iter().filter(|r| r.compared).count(),
                o.rows
                    .iter()
                    .filter(|r| r.compared)
                    .map(|r| format!("{} {}", r.name, r.model))
                    .collect::<Vec<_>>()
                    .join(", "),
            ),
        },
    };
    (f, species)
}

/// A Load that changed nothing in this long is over (refused, or its settings card cancelled).
const LOAD_GIVE_UP: std::time::Duration = std::time::Duration::from_secs(8);

/// The model in force, as (samples, epoch); `None` without a fitted one.
fn model_sig(input: &PanelInput<'_>) -> Option<(usize, u64)> {
    input.model().filter(|m| m.fitted).map(|m| (m.samples, m.epoch))
}

/// Follows a Load in flight to its end, and forgets the loaded file once the model moves on.
fn track_loads(st: &mut UiState, input: &PanelInput<'_>) {
    let now = model_sig(input);
    let card = st.model_offer.is_some();
    let b = &mut st.files;
    if let Some((path, before, since)) = b.loading.clone() {
        if now.is_some() && now != Some(before) {
            // the first change after the click: the model came in (samples or epoch moved)
            b.loading = None;
            b.loaded = now.map(|sig| (path, sig));
        } else if card {
            b.loading = Some((path, before, std::time::Instant::now()));
        } else if since.elapsed() > LOAD_GIVE_UP {
            b.loading = None;
        }
    }
    // the loaded file stays "Loaded" until the model's samples change (training, another load,
    // Forget); a refit (the epoch) is not a change — loading is followed by one
    if b.loaded.as_ref().is_some_and(|(_, sig)| now.is_none_or(|n| n.0 != sig.0)) {
        b.loaded = None;
    }
}

/// A scene card's width, in points.
const CARD_W: f32 = 168.0;

/// A quick drawing of the nuclei: tilted, sorted back to front, coloured by element.
fn preview(ui: &mut Ui, scene: Option<&Scene>, size: Vec2) {
    let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 6.0, theme::APP);
    let Some(s) = scene.filter(|s| !s.nuclei.is_empty()) else {
        p.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "empty",
            egui::FontId::proportional(14.0),
            theme::INK_FAINT,
        );
        return;
    };
    let (ca, sa) = (0.5f64.cos(), 0.5f64.sin()); // about y
    let (cb, sb) = (0.35f64.cos(), 0.35f64.sin()); // about x
    let n = s.nuclei.len() as f64;
    let c = s.nuclei.iter().fold([0.0; 3], |a, x| {
        [
            a[0] + x.pos[0] / n,
            a[1] + x.pos[1] / n,
            a[2] + x.pos[2] / n,
        ]
    });
    let mut pts: Vec<(f64, f64, f64, u8)> = s
        .nuclei
        .iter()
        .map(|x| {
            let (dx, dy, dz) = (x.pos[0] - c[0], x.pos[1] - c[1], x.pos[2] - c[2]);
            let (x1, z1) = (ca * dx + sa * dz, -sa * dx + ca * dz);
            let (y2, z2) = (cb * dy - sb * z1, sb * dy + cb * z1);
            (x1, y2, z2, x.z)
        })
        .collect();
    let extent = pts
        .iter()
        .fold(1.5f64, |m, q| m.max(q.0.abs()).max(q.1.abs()));
    let scale = (rect.width().min(rect.height()) as f64 * 0.36) / extent;
    pts.sort_by(|a, b| a.2.total_cmp(&b.2));
    for (x, y, _, z) in pts {
        let centre = rect.center() + Vec2::new((x * scale) as f32, (-y * scale) as f32);
        let r = (render::elements::radius(z) as f64 * scale * 0.9).clamp(5.0, 30.0) as f32;
        let [cr, cg, cb] = render::elements::colour_srgb(z);
        p.circle_filled(centre, r, Color32::from_rgb(cr, cg, cb));
        p.circle_filled(
            centre - Vec2::splat(r * 0.3),
            r * 0.28,
            Color32::from_white_alpha(150),
        );
    }
}

/// The window, while open.
pub fn show(ctx: &egui::Context, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    if !st.files.open {
        return;
    }
    // 82 % of the app window wide (at most 1600 points) and 86 % tall, recomputed every frame so it follows the window's size; an egui
    // Modal: centred on its measured size, the app dimmed behind it, Esc or a click outside closes
    let screen = ctx.content_rect();
    let size = Vec2::new(
        (screen.width() * 0.82).min(1600.0),
        screen.height() * 0.86,
    )
    .max(Vec2::new(360.0, 300.0));
    let inner = size - Vec2::splat(28.0);
    let mut close = false;
    let modal = egui::Modal::new(egui::Id::new("files-window"))
        .frame(
            egui::Frame::new()
                .fill(theme::PANEL)
                .stroke(Stroke::new(1.0, theme::LINE))
                .corner_radius(10)
                .inner_margin(14)
                .shadow(egui::Shadow {
                    offset: [0, 8],
                    blur: 24,
                    spread: 0,
                    color: Color32::from_black_alpha(140),
                }),
        )
        .show(ctx, |ui| {
            // exactly this size: nothing inside may widen it (a right-to-left row would take the screen)
            let (rect, _) = ui.allocate_exact_size(inner, Sense::hover());
            let mut child = ui.new_child(
                egui::UiBuilder::new()
                    .max_rect(rect)
                    .layout(egui::Layout::top_down(egui::Align::Min)),
            );
            child.set_clip_rect(rect.expand(2.0));
            let ui = &mut child;
            ui.horizontal(|ui| {
                ui.label(RichText::new("Files").size(17.0).strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .button("Close")
                        .on_hover_text("Esc, or click outside")
                        .clicked()
                    {
                        close = true;
                    }
                });
            });
            ui.separator();
            body(ui, st, input, acts);
        });
    if close || modal.should_close() {
        st.files.open = false;
    }
}

fn body(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    // ── scenes ──
    ui.horizontal(|ui| {
        theme::section(ui, "Scenes");
        let dir = crate::scene::scenes_dir()
            .map_or("scenes (created on the first save)".to_string(), |d| {
                d.display().to_string()
            });
        ui.label(
            RichText::new(dir)
                .monospace()
                .size(10.5)
                .color(theme::INK_FAINT),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .button("↻")
                .on_hover_text("read the folders again")
                .clicked()
            {
                refresh(st);
            }
        });
    });
    // the scenes take a little over half the height, the models the rest
    let total = ui.available_height();
    // room for about two and a half rows of cards (the half row says there is more), scrolled
    let row = CARD_W * 0.72 + 50.0;
    let scenes_h = st.files.scenes_h.unwrap_or((row * 2.5).min(total * 0.58)).clamp(row * 0.8, (total - 260.0).max(row * 0.8));
    egui::ScrollArea::vertical()
        .id_salt("files-scenes")
        .auto_shrink([false, false])
        .min_scrolled_height(scenes_h)
        .max_height(scenes_h)
        .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible)
        .show(ui, |ui| {
            // rows laid out here: egui's wrapping cannot size a framed card before placing it
            ui.spacing_mut().item_spacing = Vec2::splat(12.0);
            let card_outer = CARD_W + 20.0; // the frame's margins and stroke
            let cols = (((ui.available_width() - 24.0 + 12.0) / (card_outer + 12.0)).floor()
                as usize)
                .max(1);
            // the cards share the row's width: no gap at the right edge
            let card_w = ((ui.available_width() - 24.0 + 12.0) / cols as f32 - 12.0 - 20.0).max(CARD_W);
            let n = st.files.scenes.len();
            for row in (0..n).step_by(cols) {
                ui.horizontal(|ui| {
                    for i in row..(row + cols).min(n) {
                        let on = i == st.files.selected;
                        let current = st.files.scenes[i].source == st.source;
                        let r = egui::Frame::new()
                            .fill(if on { theme::ACCENT_BG } else { theme::PANEL_2 })
                            .stroke(Stroke::new(
                                1.0,
                                if on { theme::ACCENT } else { theme::LINE },
                            ))
                            .corner_radius(8)
                            .inner_margin(8)
                            .show(ui, |ui| {
                                ui.set_width(card_w);
                                ui.vertical_centered(|ui| {
                                    let e = &st.files.scenes[i];
                                    // a file that does not load says so (its hover has the error)
                                    if e.scene.is_err() {
                                        let (r, _) = ui.allocate_exact_size(Vec2::new(card_w, CARD_W * 0.72), Sense::hover());
                                        ui.painter().rect_filled(r, 6.0, theme::APP);
                                        ui.painter().text(r.center(), egui::Align2::CENTER_CENTER, "can't read", egui::FontId::proportional(14.0), theme::DANGER_INK);
                                    } else {
                                        preview(
                                        ui,
                                        e.scene.as_ref().ok(),
                                        Vec2::new(card_w, CARD_W * 0.72),
                                    );
                                    }
                                    let mut name = RichText::new(&e.name).size(15.0);
                                    if current {
                                        name = name.color(theme::ACCENT_INK).strong();
                                    }
                                    ui.label(name);
                                });
                            })
                            .response
                            .interact(Sense::click());
                        let e = &st.files.scenes[i];
                        let tip = match &e.scene {
                            Ok(s) => format!(
                                "{} · {} nuclei · {}³ × {:.3} Bohr{}",
                                e.name,
                                s.nuclei.len(),
                                s.grid_n,
                                s.grid_h,
                                if current { " · open now" } else { "" }
                            ),
                            Err(err) => format!("{}: {err}", e.name),
                        };
                        let r = r.on_hover_text(tip);
                        if r.clicked() {
                            st.files.selected = i;
                        }
                        if r.double_clicked() {
                            open_scene(st, input.scene, i, acts);
                        }
                    }
                });
            }
        });
    let sel_current = st
        .files
        .scenes
        .get(st.files.selected)
        .is_some_and(|e| e.source == st.source);
    // a scene file saved at other settings: the choice first
    confirm_card(ui, st, input.scene, acts);
    // the selected scene and its Open button, centred and large: the window's main action
    ui.add_space(6.0);
    if let Some(e) = st.files.scenes.get(st.files.selected) {
        let what = match &e.source {
            SceneSource::Preset(_) => "preset",
            SceneSource::File(_) => "scene file",
        };
        let detail = match &e.scene {
            Ok(s) => format!("{what} · {} nuclei · {}³ × {:.3} Bohr", s.nuclei.len(), s.grid_n, s.grid_h),
            Err(err) => format!("{what} · cannot be read: {err}"),
        };
        let ok = e.scene.is_ok();
        let label = if sel_current { format!("{} is open", e.name) } else { format!("Open {}", e.name) };
        ui.vertical_centered(|ui| {
            ui.label(RichText::new(detail).size(12.5).color(theme::INK_DIM));
            ui.add_space(4.0);
            let button = egui::Button::new(RichText::new(label).size(17.0).strong().color(if sel_current { theme::INK_DIM } else { Color32::WHITE }))
                .fill(if sel_current { theme::PANEL_2 } else { theme::GLOW })
                .stroke(Stroke::new(1.0, if sel_current { theme::LINE } else { theme::ACCENT }))
                .corner_radius(8)
                .min_size(Vec2::new(260.0, 44.0));
            if ui
                .add_enabled(ok && !sel_current, button)
                .on_hover_text("load it into the view (pauses; time restarts); double-clicking a card does the same")
                .clicked()
            {
                open_scene(st, input.scene, st.files.selected, acts);
            }
        });
    }

    // the splitter between the scenes and the models: drag it to share the height
    ui.add_space(2.0);
    let (bar, drag) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 10.0), Sense::drag());
    let hot = drag.hovered() || drag.dragged();
    ui.painter().line_segment(
        [egui::pos2(bar.left(), bar.center().y), egui::pos2(bar.right(), bar.center().y)],
        Stroke::new(if hot { 2.0 } else { 1.0 }, if hot { theme::ACCENT } else { theme::LINE }),
    );
    let grip = egui::Rect::from_center_size(bar.center(), Vec2::new(36.0, 4.0));
    ui.painter().rect_filled(grip, 2.0, if hot { theme::ACCENT } else { theme::PANEL_3 });
    if hot {
        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
    }
    let drag = drag.on_hover_text("drag to give the scenes or the models more room");
    if drag.dragged() {
        st.files.scenes_h = Some(scenes_h + drag.drag_delta().y);
    }
    if drag.double_clicked() {
        st.files.scenes_h = None;
    }
    // ── models ──
    ui.horizontal(|ui| {
        theme::section(ui, "Models");
        let dir = crate::scene::models_dir()
            .map_or("models (created on the first save)".to_string(), |d| {
                d.display().to_string()
            });
        ui.label(
            RichText::new(dir)
                .monospace()
                .size(10.5)
                .color(theme::INK_FAINT),
        );
    });
    let estimated = st.backend == Backend::Estimated;
    if !estimated {
        // models only run in the Estimated solver: a clear banner with the switch as its button
        egui::Frame::new()
            .fill(theme::ACCENT_BG)
            .stroke(Stroke::new(1.0, theme::ACCENT))
            .corner_radius(8)
            .inner_margin(egui::Margin::symmetric(12, 8))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.label(RichText::new("Models run in the Estimated solver").size(14.5).strong().color(theme::ACCENT_INK));
                        ui.label(RichText::new("Switch to it to load one of these, or to train your own.").size(12.5).color(theme::INK_DIM));
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let b = egui::Button::new(RichText::new("Switch to Estimated").size(15.0).strong().color(Color32::WHITE))
                            .fill(theme::GLOW)
                            .stroke(Stroke::new(1.0, theme::ACCENT))
                            .corner_radius(8)
                            .min_size(Vec2::new(200.0, 36.0));
                        if ui.add(b).on_hover_text("the toolbar's Solver ▸ Estimated").clicked() {
                            acts.push(Action::Backend(Backend::Estimated));
                        }
                    });
                });
            });
        ui.add_space(4.0);
    }
    // the open scene as it is now (its settings may have changed since it was opened), else the
    // selected one as it would load
    let scene = if sel_current {
        Some(input.scene.clone())
    } else {
        st.files.scenes.get(st.files.selected).and_then(|e| e.scene.as_ref().ok()).cloned()
    };
    track_loads(st, input);
    let rest = (ui.available_height() - 40.0).max(120.0);
    egui::ScrollArea::vertical().id_salt("files-models").auto_shrink([false, false]).max_height(rest).show(ui, |ui| {
        if st.files.models.is_empty() {
            theme::note(ui, "No models yet: train one in the Estimated solver, then File ▸ Save ▸ Model.");
        }
        let mut load: Option<PathBuf> = None;
        let mut settings_only: Option<PathBuf> = None;
        let fits: Vec<(Fit, Option<Species>)> = st
            .files
            .models
            .iter()
            .map(|m| match &scene {
                Some(s) => fit(st, input, m, s),
                None => (Fit::Unreadable("no scene selected".into()), None),
            })
            .collect();
        let best = recommended(&fits);
        for (mi, (m, (f, species))) in st.files.models.iter().zip(&fits).enumerate() {
            let (chip, colour, detail) = match &f {
                Fit::Ready => ("ready", Color32::from_rgb(0x7d, 0xd3, 0x8f), String::new()),
                Fit::Settings(_, s) => ("needs settings", Color32::from_rgb(0xf0, 0xb4, 0x5a), format!("loading applies: {s}")),
                Fit::Blocked(why) => ("can't run here", theme::DANGER_INK, why.clone()),
                Fit::Unreadable(e) => ("unreadable", theme::DANGER_INK, e.clone()),
            };
            egui::Frame::new().fill(theme::PANEL_2).corner_radius(6).inner_margin(egui::Margin::symmetric(10, 6)).show(ui, |ui| {
                ui.set_width(ui.available_width());
                // the text on the left, the Load button on the right, centred on the text's height
                const LOAD_W: f32 = 120.0;
                let text_w = (ui.available_width() - 2.0 * LOAD_W - 24.0).max(120.0);
                let text = ui
                    .allocate_ui(Vec2::new(text_w, 0.0), |ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(&m.name).strong().size(15.0));
                            theme::pill(ui, chip, colour);
                            if best == Some(mi) {
                                theme::pill(ui, "recommended", Color32::from_rgb(0x7d, 0xd3, 0x8f)).on_hover_text(
                                    "the best fit for the selected scene: it knows every element here and can run on it",
                                );
                            }
                        });
                        // the scene's elements: green = taught, red = never seen (those steps go to DFT)
                        if let Some(sp) = species {
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing.x = 4.0;
                                for &(z, taught) in &sp.scene {
                                    let sym = units::elements::symbol(z);
                                    let (bg, ink, tip) = if taught {
                                        (Color32::from_rgb(0x1f, 0x3d, 0x2a), Color32::from_rgb(0x9b, 0xe8, 0xae), format!("{sym}: taught"))
                                    } else {
                                        (Color32::from_rgb(0x45, 0x1d, 0x22), theme::DANGER_INK, format!("{sym}: never seen; its steps go to DFT"))
                                    };
                                    egui::Frame::new()
                                        .fill(bg)
                                        .corner_radius(4)
                                        .inner_margin(egui::Margin::symmetric(6, 1))
                                        .show(ui, |ui| ui.label(RichText::new(sym).size(12.5).strong().color(ink)))
                                        .response
                                        .on_hover_text(tip);
                                }
                                if !sp.covers() {
                                    ui.label(RichText::new("partly: the red ones fall back to DFT").size(12.0).color(theme::INK_FAINT));
                                }
                            });
                        }
                        let mut lines = Vec::new();
                        if !detail.is_empty() {
                            lines.push(detail);
                        }
                        if !lines.is_empty() {
                            ui.label(RichText::new(lines.join(" · ")).size(13.0).color(theme::INK_DIM));
                        }
                    })
                    .response
                    .rect;
                let busy = st.files.loading.as_ref().is_some_and(|(p, _, _)| *p == m.path);
                let in_force = st.files.loaded.as_ref().is_some_and(|(p, _)| *p == m.path);
                let can = estimated && sel_current && !input.sampling() && !busy && !in_force && st.files.loading.is_none() && matches!(f, Fit::Ready | Fit::Settings(..));
                let hover = if busy {
                    "loading…"
                } else if in_force {
                    "this model is the one in force (Load again once it has changed)"
                } else if !estimated {
                    "switch to the Estimated solver first"
                } else if !sel_current {
                    "open the selected scene first"
                } else {
                    "load this model (a settings card appears when the scene must change)"
                };
                let at = egui::Rect::from_center_size(egui::pos2(ui.max_rect().right() - LOAD_W / 2.0, text.center().y), Vec2::new(LOAD_W, 36.0));
                let word = if busy {
                    "      Loading…"
                } else if in_force {
                    "Loaded"
                } else {
                    "Load"
                };
                let button = egui::Button::new(RichText::new(word).size(15.0).strong().color(if can || busy { Color32::WHITE } else if in_force { theme::ACCENT_INK } else { theme::INK_FAINT }))
                    .fill(if can || busy { theme::GLOW } else if in_force { theme::ACCENT_BG } else { theme::PANEL_3 })
                    .stroke(Stroke::new(1.0, if can { theme::ACCENT } else { theme::LINE }))
                    .corner_radius(8);
                if ui.put(at, button.sense(if can { Sense::click() } else { Sense::hover() })).on_hover_text(hover).clicked() && can {
                    load = Some(m.path.clone());
                }
                // the model's settings without the model (to train one of your own at them)
                // (kept, disabled, once the scene has them: "Settings applied")
                if matches!(f, Fit::Settings(..) | Fit::Ready) && estimated {
                    let at2 = at.translate(Vec2::new(-(LOAD_W + 8.0), 0.0));
                    let applied = matches!(f, Fit::Ready);
                    let word = if applied { "Settings applied" } else { "Use its settings" };
                    let b2 = egui::Button::new(RichText::new(word).size(13.0).color(if applied { theme::INK_FAINT } else { theme::INK })).corner_radius(8);
                    let enabled = !applied && !input.sampling();
                    let tip = if applied {
                        "the scene already has this model's settings"
                    } else {
                        "switch the scene to this model's settings (grid, σ, charge, multiplicity, ionic potential) without loading the model"
                    };
                    if ui.put(at2, b2.sense(if enabled { Sense::click() } else { Sense::hover() })).on_hover_text(tip).clicked() && enabled {
                        settings_only = Some(m.path.clone());
                    }
                }
                if busy {
                    let spin = egui::Rect::from_center_size(egui::pos2(at.left() + 22.0, at.center().y), Vec2::splat(16.0));
                    ui.put(spin, egui::Spinner::new().size(16.0).color(Color32::WHITE));
                }
            });
        }
        if let Some(p) = settings_only {
            acts.push(Action::ApplyModelSettingsOnly(p));
        }
        if let Some(p) = load {
            st.files.loading = model_sig(input).map_or(Some((p.clone(), (usize::MAX, u64::MAX), std::time::Instant::now())), |sig| Some((p.clone(), sig, std::time::Instant::now())));
            st.model_file = Some(p);
            acts.push(Action::LoadModel);
        }
    });
    // a model anywhere else on disk: the native dialog (as the Model tab had it)
    ui.horizontal(|ui| {
        let last = st.model_file.clone().unwrap_or_else(|| crate::scene::model_path(&st.scene_path));
        if ui.add_enabled(estimated && !input.sampling(), egui::Button::new("Load from another folder…")).clicked() {
            if let Some(p) = super::dock::model::model_dialog(&last).pick_file() {
                st.model_file = Some(p);
                acts.push(Action::LoadModel);
            }
        }
    });
    if st.files.refresh_due() {
        refresh(st);
    }
}

impl Browser {
    /// A save was sent: read the folders again shortly (the worker writes the model).
    fn pending_refresh(&mut self) {
        self.refresh_at = Some(std::time::Instant::now() + std::time::Duration::from_millis(800));
    }
    fn refresh_due(&mut self) -> bool {
        if self
            .refresh_at
            .is_some_and(|t| std::time::Instant::now() >= t)
        {
            self.refresh_at = None;
            return true;
        }
        false
    }
}

fn open_scene(st: &mut UiState, current: &Scene, i: usize, acts: &mut Vec<Action>) {
    let Some(e) = st.files.scenes.get(i) else {
        return;
    };
    st.files.selected = i;
    // a scene (preset or file) that would open at another grid spacing or ionic potential than
    // the scene open now: ask first (`confirm_card`)
    if e.scene.as_ref().is_ok_and(|s| settings_differ(current, s)) {
        st.files.confirm = Some(i);
        return;
    }
    let (source, name) = (e.source.clone(), e.name.clone());
    load_entry(st, source, &name, acts);
}

/// The scene opened as it would load (a preset at the panel's quality and policy, a file as saved).
fn load_entry(st: &mut UiState, source: SceneSource, name: &str, acts: &mut Vec<Action>) {
    match source {
        SceneSource::Preset(p) => acts.push(Action::Preset(p)),
        SceneSource::File(f) => {
            st.scene_path = f;
            acts.push(Action::LoadScene { sidecar: true });
        }
    }
    st.files.scene_name = name.to_string();
}

/// The scene would open at another grid spacing or ionic potential than `current` (the open one).
fn settings_differ(current: &Scene, s: &Scene) -> bool {
    (s.grid_h - current.grid_h).abs() > 1e-9 || s.ionic != current.ionic
}

/// The choice for a scene file saved at other settings: as saved, or with the current grid spacing
/// and ionic potential (applied on the next frame, once the file is the scene: `UiState::queued`).
fn confirm_card(ui: &mut Ui, st: &mut UiState, cur: &Scene, acts: &mut Vec<Action>) {
    let Some(i) = st.files.confirm else { return };
    let Some(e) = st.files.scenes.get(i) else {
        st.files.confirm = None;
        return;
    };
    let Ok(s) = e.scene.as_ref() else {
        st.files.confirm = None;
        return;
    };
    let (name, saved, current) = (
        e.name.clone(),
        format!("{}³ × {:.3} Bohr, ionic potential {}", s.grid_n, s.grid_h, s.ionic.word()),
        format!("{}³ × {:.3} Bohr, ionic potential {}", cur.grid_n, cur.grid_h, cur.ionic.word()),
    );
    let (source, keep_h, keep_ionic) = (e.source.clone(), cur.grid_h, cur.ionic);
    egui::Frame::new()
        .fill(theme::ACCENT_BG)
        .stroke(Stroke::new(1.0, theme::ACCENT))
        .corner_radius(8)
        .inner_margin(egui::Margin::symmetric(14, 10))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new(format!("Open {name} with which settings?")).strong().size(14.5).color(theme::ACCENT_INK));
            let what = if matches!(source, SceneSource::Preset(_)) { "opens at" } else { "saved at" };
            ui.label(RichText::new(format!("{what} {saved}\nthe open scene: {current}")).size(12.5).color(theme::INK_DIM));
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let big = |t: &str| egui::Button::new(RichText::new(t).strong().size(14.0).color(Color32::WHITE)).fill(theme::GLOW).corner_radius(6).min_size(Vec2::new(200.0, 32.0));
                if ui.add(big("Use the scene's settings")).on_hover_text("open it as it was saved (a preset: at the app's grid and ionic potential)").clicked() {
                    load_entry(st, source.clone(), &name, acts);
                    st.files.confirm = None;
                }
                if ui.add(big("Keep my settings")).on_hover_text("open it, then put it on your current grid spacing and ionic potential (the nuclei stay where they are)").clicked() {
                    load_entry(st, source.clone(), &name, acts);
                    st.queued.push(Action::Quality(keep_h));
                    st.queued.push(Action::Ionic(keep_ionic));
                    st.files.confirm = None;
                }
                if ui.button("Cancel").clicked() {
                    st.files.confirm = None;
                }
            });
        });
}

/// A file name from what was typed: path separators and reserved characters become '_'.
fn clean(name: &str) -> String {
    let s: String = name
        .trim()
        .chars()
        .map(|c| if "\\/:*?\"<>|".contains(c) { '_' } else { c })
        .collect();
    if s.is_empty() {
        "untitled".into()
    } else {
        s
    }
}

/// What the Save window writes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum SaveKind {
    #[default]
    Scene,
    Model,
    Both,
}

/// Opens the Save window (`File ▸ Save…`), named after the scene in force.
pub fn open_save(st: &mut UiState) {
    let name = match &st.source {
        SceneSource::Preset(p) => p.clone(),
        SceneSource::File(f) => stem(Path::new(f)),
    };
    let b = &mut st.files;
    b.save_open = true;
    b.save_name = name;
}

/// One of the three big choices: a drawing and a word.
fn kind_button(ui: &mut Ui, on: bool, enabled: bool, label: &str, draw: impl FnOnce(&egui::Painter, egui::Rect)) -> egui::Response {
    let (rect, r) = ui.allocate_exact_size(Vec2::new(150.0, 118.0), if enabled { Sense::click() } else { Sense::hover() });
    let p = ui.painter();
    let (fill, stroke) = if on {
        (theme::ACCENT_BG, theme::ACCENT)
    } else if r.hovered() && enabled {
        (theme::PANEL_3, theme::LINE)
    } else {
        (theme::PANEL_2, theme::LINE)
    };
    p.rect(rect, 10.0, fill, Stroke::new(1.0, stroke), egui::StrokeKind::Inside);
    let art = egui::Rect::from_center_size(rect.center() - Vec2::new(0.0, 12.0), Vec2::new(110.0, 62.0));
    draw(p, art);
    let ink = if !enabled {
        theme::INK_FAINT
    } else if on {
        theme::ACCENT_INK
    } else {
        theme::INK
    };
    p.text(egui::pos2(rect.center().x, rect.bottom() - 16.0), egui::Align2::CENTER_CENTER, label, egui::FontId::proportional(15.0), ink);
    if !enabled {
        p.rect_filled(rect, 10.0, theme::PANEL.gamma_multiply(0.55));
    }
    r
}

/// The current scene's nuclei as a small glyph (the scene choice's drawing).
fn scene_glyph(p: &egui::Painter, rect: egui::Rect, scene: &Scene) {
    let n = scene.nuclei.len().max(1) as f64;
    let c = scene.nuclei.iter().fold([0.0; 2], |a, x| [a[0] + x.pos[0] / n, a[1] + x.pos[1] / n]);
    let extent = scene.nuclei.iter().fold(1.5f64, |m, x| m.max((x.pos[0] - c[0]).abs()).max((x.pos[1] - c[1]).abs()));
    let scale = (rect.height() as f64 * 0.38) / extent;
    for x in &scene.nuclei {
        let at = rect.center() + Vec2::new(((x.pos[0] - c[0]) * scale) as f32, (-(x.pos[1] - c[1]) * scale) as f32);
        let [r, g, b] = render::elements::colour_srgb(x.z);
        p.circle_filled(at, (render::elements::radius(x.z) as f64 * scale).clamp(4.0, 14.0) as f32, Color32::from_rgb(r, g, b));
    }
}

/// The Save window, while open.
pub fn show_save(ctx: &egui::Context, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    if !st.files.save_open {
        return;
    }
    let estimated = st.backend == Backend::Estimated;
    let fitted = estimated && input.model().is_some_and(|m| m.fitted);
    if !fitted && st.files.save_kind != SaveKind::Scene {
        st.files.save_kind = SaveKind::Scene; // no model to save
    }
    let mut close = false;
    let modal = egui::Modal::new(egui::Id::new("save-window"))
        .frame(
            egui::Frame::new()
                .fill(theme::PANEL)
                .stroke(Stroke::new(1.0, theme::LINE))
                .corner_radius(10)
                .inner_margin(18)
                .shadow(egui::Shadow { offset: [0, 8], blur: 24, spread: 0, color: Color32::from_black_alpha(140) }),
        )
        .show(ctx, |ui| {
            ui.set_width(150.0 * 3.0 + 2.0 * 12.0);
            ui.label(RichText::new("Save").size(17.0).strong());
            ui.add_space(8.0);
            let model_tip = "no fitted model: train one in the Estimated solver first";
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 12.0;
                let k = st.files.save_kind;
                if kind_button(ui, k == SaveKind::Scene, true, "Scene", |p, r| scene_glyph(p, r, input.scene))
                    .on_hover_text("the nuclei, their velocities and the grid, as they are now")
                    .clicked()
                {
                    st.files.save_kind = SaveKind::Scene;
                }
                let r = kind_button(ui, k == SaveKind::Model, fitted, "Model", |p, r| {
                    super::icons::paint(p, egui::Rect::from_center_size(r.center(), Vec2::splat(54.0)), super::icons::Icon::Model, theme::ACCENT_INK)
                });
                if r.on_hover_text(if fitted { "the estimated solver's model, its fit and its samples" } else { model_tip }).clicked() && fitted {
                    st.files.save_kind = SaveKind::Model;
                }
                let r = kind_button(ui, k == SaveKind::Both, fitted, "Both", |p, r| {
                    let (a, b) = r.split_left_right_at_fraction(0.5);
                    scene_glyph(p, a, input.scene);
                    super::icons::paint(p, egui::Rect::from_center_size(b.center(), Vec2::splat(44.0)), super::icons::Icon::Model, theme::ACCENT_INK);
                });
                if r.on_hover_text(if fitted { "the scene and its model under one name: name.scene and name.model, together" } else { model_tip }).clicked() && fitted {
                    st.files.save_kind = SaveKind::Both;
                }
            });
            ui.add_space(12.0);
            let kind = st.files.save_kind;
            ui.horizontal(|ui| {
                ui.label("Name");
                ui.add(egui::TextEdit::singleline(&mut st.files.save_name).desired_width(260.0).hint_text("name"));
            });
            ui.add_space(4.0);
            let folder_word = match kind {
                SaveKind::Scene => "the scenes folder",
                SaveKind::Model => "the models folder",
                SaveKind::Both => "the scenes and models folders",
            };
            ui.radio_value(&mut st.files.save_elsewhere, false, format!("In {folder_word}"));
            ui.radio_value(&mut st.files.save_elsewhere, true, "Somewhere else… (choose in the next step)");
            let name = clean(&st.files.save_name);
            let scene_dir = crate::scene::scenes_dir().unwrap_or_else(|| PathBuf::from("scenes"));
            let model_dir = crate::scene::models_dir().unwrap_or_else(|| PathBuf::from("models"));
            let scene_file = scene_dir.join(format!("{name}.scene"));
            let model_file = model_dir.join(format!("{name}.model"));
            // in the folders: a name already taken is refused for Both, replaced on confirmation otherwise
            let mut taken: Vec<String> = Vec::new();
            if !st.files.save_elsewhere {
                if kind != SaveKind::Model && scene_file.exists() {
                    taken.push(format!("{name}.scene"));
                }
                if kind != SaveKind::Scene && model_file.exists() {
                    taken.push(format!("{name}.model"));
                }
            }
            let blocked = kind == SaveKind::Both && !taken.is_empty();
            if !taken.is_empty() {
                let (words, colour) = if blocked {
                    (format!("{} already exists: pick another name to save both together", taken.join(" and ")), theme::DANGER_INK)
                } else {
                    (format!("{} already exists: Save replaces it", taken.join(" and ")), Color32::from_rgb(0xf0, 0xb4, 0x5a))
                };
                ui.label(RichText::new(words).size(12.5).color(colour));
            }
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let label = if !taken.is_empty() && !blocked {
                        "Replace"
                    } else if st.files.save_elsewhere {
                        "Choose where…"
                    } else {
                        "Save"
                    };
                    let can = !blocked && !st.files.save_name.trim().is_empty();
                    if ui.add_enabled(can, egui::Button::new(RichText::new(label).strong()).min_size(Vec2::new(110.0, 30.0))).clicked() {
                        let (scene_to, model_to) = if st.files.save_elsewhere {
                            // one dialog: the files land side by side under the chosen name
                            let ext = if kind == SaveKind::Model { "model" } else { "scene" };
                            let what = if kind == SaveKind::Model { "Atomic Sandbox model" } else { "Atomic Sandbox scene" };
                            match rfd::FileDialog::new().add_filter(what, &[ext]).set_file_name(format!("{name}.{ext}")).save_file() {
                                Some(p) => (Some(p.with_extension("scene")), Some(p.with_extension("model"))),
                                None => (None, None),
                            }
                        } else {
                            if kind != SaveKind::Model {
                                let _ = std::fs::create_dir_all(&scene_dir);
                            }
                            if kind != SaveKind::Scene {
                                let _ = std::fs::create_dir_all(&model_dir);
                            }
                            (Some(scene_file.clone()), Some(model_file.clone()))
                        };
                        if let (Some(sp), Some(mp)) = (scene_to, model_to) {
                            if kind != SaveKind::Model {
                                st.scene_path = sp.display().to_string();
                                acts.push(Action::SaveScene);
                            }
                            if kind != SaveKind::Scene {
                                st.model_file = Some(mp);
                                acts.push(Action::SaveModel);
                            }
                            st.files.pending_refresh();
                            close = true;
                        }
                    }
                    if ui.button("Cancel").clicked() {
                        close = true;
                    }
                });
            });
        });
    if close || modal.should_close() {
        st.files.save_open = false;
    }
}

/// The Save window on Model (the Model tab's Save model…).
pub fn open_save_model(st: &mut UiState) {
    open_save(st);
    st.files.save_kind = SaveKind::Model;
}
