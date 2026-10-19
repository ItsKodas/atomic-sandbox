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
    model_name: String,
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

/// Where the model stands against a scene.
enum Fit {
    Ready,
    /// Loads after the card applies these settings.
    Settings(String),
    Blocked(String),
    Unreadable(String),
}

fn fit(
    st: &UiState,
    input: &PanelInput<'_>,
    m: &ModelEntry,
    scene: &Scene,
) -> (Fit, Option<String>) {
    let prov = match &m.prov {
        Ok(p) => p,
        Err(e) => return (Fit::Unreadable(e.clone()), None),
    };
    // the elements: the model answers only the species it was taught (the rest fall back to DFT)
    let taught: Vec<u8> = prov.masses.iter().map(|&(z, _)| z).collect();
    let mut missing: Vec<u8> = scene
        .nuclei
        .iter()
        .map(|n| n.z)
        .filter(|z| !taught.contains(z))
        .collect();
    missing.sort_unstable();
    missing.dedup();
    let species = if taught.is_empty() {
        None
    } else if missing.is_empty() {
        Some(format!("taught {}", symbols(&taught)))
    } else {
        Some(format!(
            "never saw {} (those steps go to DFT)",
            symbols(&missing)
        ))
    };
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

fn symbols(zs: &[u8]) -> String {
    let mut zs = zs.to_vec();
    zs.sort_unstable();
    zs.dedup();
    zs.iter()
        .map(|&z| units::elements::symbol(z))
        .collect::<Vec<_>>()
        .join(", ")
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
    // 80 % of the app window, recomputed every frame so it follows the window's size; an egui
    // Modal: centred on its measured size, the app dimmed behind it, Esc or a click outside closes
    let screen = ctx.content_rect();
    let size = Vec2::new(
        (screen.width() * 0.8).min(1280.0),
        (screen.height() * 0.8).min(900.0),
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
    let scenes_h = (row * 2.5).min(total * 0.58).max(row * 1.2);
    egui::ScrollArea::vertical()
        .id_salt("files-scenes")
        .auto_shrink([false, false])
        .min_scrolled_height(scenes_h)
        .max_height(scenes_h)
        .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible)
        .show(ui, |ui| {
            // rows laid out here: egui's wrapping cannot size a framed card before placing it
            ui.spacing_mut().item_spacing = Vec2::splat(12.0);
            let card_outer = CARD_W + 18.0; // the frame's margins and stroke
            let cols = (((ui.available_width() - 14.0 + 12.0) / (card_outer + 12.0)).floor()
                as usize)
                .max(1);
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
                                ui.set_width(CARD_W);
                                ui.vertical_centered(|ui| {
                                    let e = &st.files.scenes[i];
                                    preview(
                                        ui,
                                        e.scene.as_ref().ok(),
                                        Vec2::new(CARD_W, CARD_W * 0.72),
                                    );
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
                            open_scene(st, i, acts);
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
    ui.horizontal(|ui| {
        if let Some(e) = st.files.scenes.get(st.files.selected) {
            let what = match &e.source {
                SceneSource::Preset(_) => "preset",
                SceneSource::File(_) => "file",
            };
            ui.label(RichText::new(format!("{} ({what})", e.name)).strong());
            let ok = e.scene.is_ok();
            if ui
                .add_enabled(ok && !sel_current, egui::Button::new(if sel_current { "Open now" } else { "Open scene" }))
                .on_hover_text("load it into the view (pauses; time restarts); double-click a card does the same")
                .clicked()
            {
                open_scene(st, st.files.selected, acts);
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("Save current scene").on_hover_text("positions and velocities as they are now, into the scenes folder").clicked() {
                let dir = crate::scene::scenes_dir().unwrap_or_else(|| PathBuf::from("scenes"));
                if std::fs::create_dir_all(&dir).is_ok() {
                    st.scene_path = dir.join(format!("{}.scene", clean(&st.files.scene_name))).display().to_string();
                    acts.push(Action::SaveScene);
                    st.files.pending_refresh();
                }
            }
            ui.add(egui::TextEdit::singleline(&mut st.files.scene_name).desired_width(130.0).hint_text("name"));
        });
    });

    ui.add_space(6.0);
    ui.separator();
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
        ui.horizontal(|ui| {
            theme::note(ui, "Models run in the Estimated solver.");
            if ui.small_button("Switch to Estimated").clicked() {
                acts.push(Action::Backend(Backend::Estimated));
            }
        });
    }
    let scene = st
        .files
        .scenes
        .get(st.files.selected)
        .and_then(|e| e.scene.as_ref().ok())
        .cloned();
    let rest = (ui.available_height() - 80.0).max(120.0);
    egui::ScrollArea::vertical().id_salt("files-models").auto_shrink([false, false]).max_height(rest).show(ui, |ui| {
        if st.files.models.is_empty() {
            theme::note(ui, "No models yet: train one in the Estimated solver, then save it below.");
        }
        let mut load: Option<PathBuf> = None;
        for m in &st.files.models {
            let (f, species) = match &scene {
                Some(s) => fit(st, input, m, s),
                None => (Fit::Unreadable("no scene selected".into()), None),
            };
            let (chip, colour, detail) = match &f {
                Fit::Ready => ("ready", Color32::from_rgb(0x7d, 0xd3, 0x8f), String::new()),
                Fit::Settings(s) => ("needs settings", Color32::from_rgb(0xf0, 0xb4, 0x5a), format!("loading applies: {s}")),
                Fit::Blocked(why) => ("can't run here", theme::DANGER_INK, why.clone()),
                Fit::Unreadable(e) => ("unreadable", theme::DANGER_INK, e.clone()),
            };
            egui::Frame::new().fill(theme::PANEL_2).corner_radius(6).inner_margin(egui::Margin::symmetric(10, 6)).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&m.name).strong().size(15.0));
                    theme::pill(ui, chip, colour);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let can = estimated && sel_current && !input.sampling() && matches!(f, Fit::Ready | Fit::Settings(_));
                        let hover = if !estimated {
                            "switch to the Estimated solver first"
                        } else if !sel_current {
                            "open the selected scene first"
                        } else {
                            "load this model (a settings card appears when the scene must change)"
                        };
                        if ui.add_enabled(can, egui::Button::new("Load")).on_hover_text(hover).on_disabled_hover_text(hover).clicked() {
                            load = Some(m.path.clone());
                        }
                    });
                });
                let mut lines = Vec::new();
                if let Some(s) = &species {
                    lines.push(s.clone());
                }
                if !detail.is_empty() {
                    lines.push(detail);
                }
                if !lines.is_empty() {
                    ui.label(RichText::new(lines.join(" · ")).size(13.0).color(theme::INK_DIM));
                }
            });
        }
        if let Some(p) = load {
            st.model_file = Some(p);
            acts.push(Action::LoadModel);
        }
    });
    let fitted = input.model().is_some_and(|m| m.fitted);
    ui.horizontal(|ui| {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .add_enabled(estimated && fitted, egui::Button::new("Save current model"))
                .on_hover_text("the model, its fit and its samples, into the models folder")
                .on_disabled_hover_text("a fitted model in the Estimated solver")
                .clicked()
            {
                let dir = crate::scene::models_dir().unwrap_or_else(|| PathBuf::from("models"));
                if std::fs::create_dir_all(&dir).is_ok() {
                    let name = if st.files.model_name.trim().is_empty() {
                        st.files.scene_name.clone()
                    } else {
                        st.files.model_name.clone()
                    };
                    st.model_file = Some(dir.join(format!("{}.model", clean(&name))));
                    acts.push(Action::SaveModel);
                    st.files.pending_refresh();
                }
            }
            ui.add(
                egui::TextEdit::singleline(&mut st.files.model_name)
                    .desired_width(130.0)
                    .hint_text("name"),
            );
        });
    });
    // anywhere else on disk: the native dialogs (as the Model tab had them)
    ui.horizontal(|ui| {
        let last = st
            .model_file
            .clone()
            .unwrap_or_else(|| crate::scene::model_path(&st.scene_path));
        if ui
            .add_enabled(
                estimated && !input.sampling(),
                egui::Button::new("Load from another folder…"),
            )
            .clicked()
        {
            if let Some(p) = super::dock::model::model_dialog(&last).pick_file() {
                st.model_file = Some(p);
                acts.push(Action::LoadModel);
            }
        }
        if ui
            .add_enabled(
                estimated && fitted,
                egui::Button::new("Save to another folder…"),
            )
            .clicked()
        {
            if let Some(p) = super::dock::model::model_dialog(&last).save_file() {
                st.model_file = Some(super::dock::model::with_model_extension(p));
                acts.push(Action::SaveModel);
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

fn open_scene(st: &mut UiState, i: usize, acts: &mut Vec<Action>) {
    let Some(e) = st.files.scenes.get(i) else {
        return;
    };
    st.files.selected = i;
    match e.source.clone() {
        SceneSource::Preset(p) => acts.push(Action::Preset(p)),
        SceneSource::File(f) => {
            st.scene_path = f;
            acts.push(Action::LoadScene { sidecar: true });
        }
    }
    st.files.scene_name = e.name.clone();
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
