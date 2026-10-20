//! The capture buttons at the top right of the 3D view (`crate::capture`): Screenshot, Record /
//! Stop and the quick settings popover (hide the UI, the labels and info over a capture, the
//! frame's resolution and aspect with its guide over the view, the capture camera — the view's, a
//! fixed angle or an orbit, drawn in the view as a gizmo —, record mode, fs per frame). While
//! recording, a red dot with the frames, the simulated time covered and the video's length; after
//! it, the finish's state and the last note. The popover's first button opens the pop-out view
//! (`crate::capture::popout`, `Settings::popout`). Screenshot and Record go through [`Action`]; the
//! settings are display state written to `UiState::capture` directly.
use super::{icons, theme, Action, UiState, RED};
use crate::capture::camera::{self, Aspect, CameraMode, CameraSettings};
use crate::capture::{fs_words, RecordMode, Resolution};
use egui::{pos2, Align2, Id, Order, Rect, RichText, Stroke, Vec2};
use std::time::Duration;

/// How long the last note stays under the buttons.
const NOTE_FOR: Duration = Duration::from_secs(8);

/// Draws the buttons (top right of `view`) and returns the bottom of what they took, for the cards
/// stacked under them (the atom card).
pub fn show(ctx: &egui::Context, view: Rect, st: &mut UiState, camera: &render::camera::OrbitCamera, acts: &mut Vec<Action>) -> f32 {
    let cv = st.capture_view.clone();
    let recording = cv.recording.is_some();
    let time = ctx.input(|i| i.time);
    let area = egui::Area::new(Id::new("capture-buttons"))
        .pivot(Align2::RIGHT_TOP)
        .fixed_pos(pos2(view.right() - 12.0, view.top() + 12.0))
        .order(Order::Middle)
        .fade_in(false)
        .show(ctx, |ui| {
            theme::card().inner_margin(egui::Margin::symmetric(6, 5)).show(ui, |ui| {
                ui.spacing_mut().item_spacing = Vec2::new(4.0, 4.0);
                ui.horizontal(|ui| {
                    if let Some(r) = &cv.recording {
                        // the red dot, blinking once a second
                        let on = time.fract() < 0.6;
                        let (rect, _) = ui.allocate_exact_size(Vec2::splat(12.0), egui::Sense::hover());
                        ui.painter().circle_filled(rect.center(), 5.0, if on { RED } else { RED.gamma_multiply(0.35) });
                        let what = match r.mode {
                            RecordMode::SimTime => format!("{} · {} frames · {}", fs_words(r.sim_fs), r.frames, video_len(r)),
                            RecordMode::AsSeen => format!("{} frames · {}", r.frames, video_len(r)),
                        };
                        ui.label(RichText::new(what).monospace().size(11.5).color(theme::INK));
                        ui.ctx().request_repaint_after(Duration::from_millis(100));
                    }
                    let shot = icons::button(ui, icons::Icon::Camera, 28.0, false, true)
                        .on_hover_text(format!("Screenshot → captures/ ({})", if st.capture.hide_ui || !cv.surface_copyable { "3D view only" } else { "the whole window" }));
                    if shot.clicked() {
                        acts.push(Action::Screenshot);
                    }
                    let rec_icon = if recording { icons::Icon::Stop } else { icons::Icon::Record };
                    let rec = ui.add(egui::Button::selectable(recording, "").min_size(Vec2::splat(28.0)));
                    let fg = if recording { theme::INK } else { RED };
                    icons::paint(ui.painter(), Rect::from_center_size(rec.rect.center(), Vec2::splat(15.0)), rec_icon, fg);
                    let rec = rec.on_hover_text(if recording {
                        "Stop recording (the video is encoded with ffmpeg when it is on PATH)".to_string()
                    } else {
                        match st.capture.mode {
                            RecordMode::SimTime => format!("Record by simulated time: one frame per {}, played at 30 fps", fs_words(st.capture.fs_per_frame)),
                            RecordMode::AsSeen => "Record as seen: 30 frames per second of wall time".to_string(),
                        }
                    });
                    if rec.clicked() {
                        acts.push(Action::Record(!recording));
                    }
                    let gear = icons::button(ui, icons::Icon::Sliders, 28.0, st.capture.open, true).on_hover_text("Capture settings");
                    if gear.clicked() {
                        st.capture.open = !st.capture.open;
                    }
                });
            });
            let rect = ui.min_rect();
            super::search::mark(ui, st, "v-capture", rect);
        });
    let rect = area.response.rect;
    // the notes in their own box under the buttons (a long one wraps there instead of widening
    // the button card): the recording's warnings, the finish's state, the last result
    let mut notes: Vec<(String, egui::Color32)> = cv.recording.iter().flat_map(|r| r.notes.iter().map(|n| (n.clone(), super::AMBER))).collect();
    if let Some(f) = &cv.finishing {
        notes.push((f.clone(), theme::INK_DIM));
        ctx.request_repaint_after(Duration::from_millis(200));
    } else if let Some((_, text, err)) = cv.last.as_ref().filter(|l| l.0.elapsed() < NOTE_FOR) {
        notes.push((text.clone(), if *err { RED } else { theme::INK_DIM }));
        ctx.request_repaint_after(Duration::from_millis(500));
    }
    let mut bottom = rect.bottom();
    if !notes.is_empty() {
        let r = egui::Area::new(Id::new("capture-notes"))
            .pivot(Align2::RIGHT_TOP)
            .fixed_pos(pos2(rect.right(), rect.bottom() + 6.0))
            .order(Order::Middle)
            .interactable(false)
            .fade_in(false)
            .show(ctx, |ui| {
                theme::card().inner_margin(egui::Margin::symmetric(8, 5)).show(ui, |ui| {
                    ui.set_max_width(NOTES_WIDTH);
                    ui.spacing_mut().item_spacing.y = 3.0;
                    for (text, colour) in &notes {
                        ui.add(egui::Label::new(RichText::new(text).size(11.0).color(*colour)).wrap());
                    }
                });
            });
        bottom = r.response.rect.bottom();
    }
    if st.capture.open {
        let anchor = Rect::from_min_max(rect.min, pos2(rect.right(), bottom));
        bottom = settings(ctx, view, anchor, st, &cv, camera).max(bottom);
    }
    guide(ctx, view, st, &cv);
    bottom
}

/// "4.1 s video", or "4.1 / 10.0 s video" with a set length.
fn video_len(r: &crate::capture::RecordingView) -> String {
    match r.limit_s {
        Some(l) => format!("{:.1} / {l:.1} s video", r.playback_s),
        None => format!("{:.1} s video", r.playback_s),
    }
}

/// The notes box's width at most (points).
const NOTES_WIDTH: f32 = 260.0;

/// While the settings are open or a recording runs, a hidden-UI capture from the view's camera at
/// another aspect: the frame it takes, outlined over the view and the rest dimmed.
fn guide(ctx: &egui::Context, view: Rect, st: &UiState, cv: &crate::capture::View) {
    let s = &st.capture;
    let rec = cv.recording.as_ref();
    let hidden = s.hide_ui || rec.is_some_and(|r| r.hide_ui);
    if !(s.open || rec.is_some()) || !hidden || s.camera.mode != CameraMode::View {
        return;
    }
    let ppp = ctx.pixels_per_point();
    let view_px = ((view.width() * ppp) as u32, (view.height() * ppp) as u32);
    let (w, h) = rec.map_or_else(|| s.frame_size(view_px, cv.max_side), |r| r.size);
    if w == 0 || h == 0 {
        return;
    }
    let ratio = w as f32 / h as f32;
    if (ratio - view.width() / view.height()).abs() < 0.01 {
        return;
    }
    let (x, y, gw, gh) = camera::inscribed((view.left(), view.top(), view.width(), view.height()), ratio);
    let g = Rect::from_min_size(pos2(x, y), egui::vec2(gw, gh));
    let p = ctx.layer_painter(egui::LayerId::new(Order::Background, Id::new("capture-guide"))).with_clip_rect(view);
    let dim = egui::Color32::from_black_alpha(110);
    for r in [
        Rect::from_min_max(view.min, pos2(view.right(), g.top())),
        Rect::from_min_max(pos2(view.left(), g.bottom()), view.max),
        Rect::from_min_max(pos2(view.left(), g.top()), pos2(g.left(), g.bottom())),
        Rect::from_min_max(pos2(g.right(), g.top()), pos2(view.right(), g.bottom())),
    ] {
        if r.is_positive() {
            p.rect_filled(r, 0.0, dim);
        }
    }
    p.rect_stroke(g, 0.0, Stroke::new(1.5, AMBER_GUIDE), egui::StrokeKind::Inside);
    p.text(g.left_bottom() + egui::vec2(8.0, -6.0), Align2::LEFT_BOTTOM, format!("capture {}", s.aspect.label()), egui::FontId::proportional(11.0), AMBER_GUIDE);
}

/// The guide frame's and the camera gizmo's colour.
const AMBER_GUIDE: egui::Color32 = egui::Color32::from_rgb(255, 184, 71);

/// The quick settings popover under `anchor` (the buttons and their notes), until the settings
/// button (or Esc over it) closes it. Returns its bottom.
fn settings(ctx: &egui::Context, view: Rect, anchor: Rect, st: &mut UiState, cv: &crate::capture::View, window: &render::camera::OrbitCamera) -> f32 {
    let recording = cv.recording.is_some();
    let ppp = ctx.pixels_per_point();
    let view_px = ((view.width() * ppp) as u32, (view.height() * ppp) as u32);
    let max_h = (view.bottom() - anchor.bottom() - 24.0).max(160.0);
    let r = egui::Area::new(Id::new("capture-settings"))
        .pivot(Align2::RIGHT_TOP)
        .fixed_pos(pos2(anchor.right(), anchor.bottom() + 6.0))
        .order(Order::Foreground)
        .fade_in(false)
        .show(ctx, |ui| {
            theme::card().fill(theme::PANEL).stroke(Stroke::new(1.0, theme::LINE)).show(ui, |ui| {
                egui::ScrollArea::vertical().max_height(max_h).show(ui, |ui| {
                    ui.set_width(320.0);
                    // the pop-out view (`crate::capture::popout`): a window of its own showing what a capture takes
                    let open = st.capture.popout;
                    let pop = ui
                        .add(egui::Button::selectable(open, if open { "Close the pop-out view" } else { "Pop out view" }).min_size(Vec2::new(ui.available_width(), 24.0)))
                        .on_hover_text("A separate window showing exactly what Screenshot and Record take now: the frame's size and aspect, the capture camera, the look and the labels over it — and a recording's red dot and frame count. Move it to another monitor, resize or maximise it; F11 there for full screen. Closing it changes nothing else.");
                    if pop.clicked() {
                        st.capture.popout = !open;
                    }
                    super::search::mark(ui, st, "v-capture-popout", pop.rect);
                    ui.add_space(2.0);
                    let st_camera_view = st.camera_view;
                    let s = &mut st.capture;
                    theme::section(ui, "Capture");
                    let can_show_ui = cv.surface_copyable;
                    ui.add_enabled_ui(can_show_ui, |ui| {
                        ui.checkbox(&mut s.hide_ui, "Hide the UI (the 3D view only)")
                            .on_hover_text("No chrome and no editor geometry: the cell box, walls, floor grid, axes, placement grid, selection highlight and bond-angle arcs are left out. Bonds, trails and the cloud stay.");
                    });
                    if !can_show_ui {
                        theme::note(ui, "this window's surface cannot be copied: captures are the 3D view only");
                    }
                    let hidden = s.hide_ui || !can_show_ui;
                    ui.add_enabled_ui(hidden, |ui| {
                        ui.checkbox(&mut s.overlay.on, "Show labels and info over it")
                            .on_hover_text("Drawn into the capture: the view's labels, a title with the molecules, the time and the rate, and a distance scale.");
                        if s.overlay.on {
                            ui.indent("capture-overlay", |ui| {
                                ui.checkbox(&mut s.overlay.labels, "labels (bond lengths, angles, charges)");
                                ui.checkbox(&mut s.overlay.title, "title: the molecules' formulas and names")
                                    .on_hover_text("Top left: the molecules as the view names them (View ▸ Bonds and molecules ▸ name style), instead of their names over them, with the scene and the solver.");
                                ui.checkbox(&mut s.overlay.time, "time, step and rate");
                                ui.checkbox(&mut s.overlay.scale, "distance scale (at the camera's focus)");
                            });
                        }
                    });
                    ui.add_space(4.0);
                    theme::section(ui, "Frame");
                    ui.add_enabled_ui(hidden, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.label("Resolution");
                            for res in Resolution::ALL {
                                if ui.add(egui::Button::selectable(s.resolution == res, res.label())).clicked() {
                                    s.resolution = res;
                                }
                            }
                        });
                        ui.horizontal_wrapped(|ui| {
                            ui.label("Aspect");
                            for a in Aspect::ALL {
                                if ui.add(egui::Button::selectable(s.aspect == a, a.label())).clicked() {
                                    s.aspect = a;
                                }
                            }
                        });
                        let (w, h) = s.frame_size(view_px, cv.max_side);
                        let asked = s.frame_size(view_px, u32::MAX);
                        if (w, h) == asked {
                            theme::note(ui, format!("{w} × {h} pixels"));
                        } else {
                            theme::note(ui, format!("{w} × {h} pixels ({} × {} is beyond the GPU's {} px limit{})", asked.0, asked.1, cv.max_side, if st_camera_view { " in the Camera view" } else { "" }));
                        }
                    });
                    if !hidden {
                        theme::note(ui, "with the UI shown the frame is the window, at its size");
                    }
                    ui.add_space(4.0);
                    theme::section(ui, "Camera");
                    ui.add_enabled_ui(hidden, |ui| camera_controls(ui, &mut s.camera, window));
                    ui.add_space(4.0);
                    theme::section(ui, "Record");
                    if let Some(m) = theme::segmented(
                        ui,
                        s.mode,
                        &[
                            (RecordMode::SimTime, "simulated time", "One frame per slice of simulated time, played at 30 fps: a slow DFT run becomes a smooth time-lapse. Nothing is recorded while paused."),
                            (RecordMode::AsSeen, "as seen", "30 frames per second of wall time, as on screen (for recording interaction)."),
                        ],
                    ) {
                        s.mode = m;
                    }
                    if s.mode == RecordMode::SimTime {
                        ui.horizontal(|ui| {
                            ui.label("fs per frame");
                            ui.add(egui::DragValue::new(&mut s.fs_per_frame).range(0.05..=100.0).speed(0.01).max_decimals(2).suffix(" fs"));
                        });
                        let per_ps = 1000.0 / s.fs_per_frame.max(1e-3) / crate::capture::clock::FPS;
                        theme::note(ui, format!("1 ps of simulation → {per_ps:.1} s of video; an O–H stretch (≈ 9 fs) gets {:.0} frames", 9.0 / s.fs_per_frame.max(1e-3)));
                        theme::note(ui, "sim-time frames show the 3D view only (the UI would show wall-clock state)");
                    }
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut s.stop_on, "stop at")
                            .on_hover_text("Stop by itself once the video is this long, however long the run takes to get there (a paused run records nothing in simulated time)");
                        ui.add_enabled(s.stop_on, egui::DragValue::new(&mut s.stop_after_s).range(0.5..=3600.0).speed(0.5).max_decimals(1).suffix(" s of video"));
                    });
                    if s.stop_on && s.mode == RecordMode::SimTime {
                        theme::note(ui, format!("{:.0} frames = {} of simulation at {} per frame", (s.stop_after_s * crate::capture::clock::FPS).round(), crate::capture::fs_words(s.stop_after_s * crate::capture::clock::FPS * s.fs_per_frame), fs_words(s.fs_per_frame)));
                    }
                    if recording {
                        theme::note(ui, "the frame and the record mode apply to the next recording; the camera and the labels change now");
                    }
                    ui.add_space(4.0);
                    match cv.ffmpeg {
                        Some(true) => theme::note(ui, "ffmpeg found: recordings are encoded to MP4 (H.264, 30 fps)"),
                        Some(false) => theme::note(ui, "ffmpeg not on PATH: recordings stay PNG sequences (the note says how to encode)"),
                        None => theme::note(ui, "looking for ffmpeg…"),
                    };
                    ui.horizontal(|ui| {
                        theme::note(ui, "files go to captures/");
                        if ui.small_button("open folder").clicked() {
                            let dir = crate::capture::captures_dir();
                            std::fs::create_dir_all(&dir).ok();
                            open_folder(&dir);
                        }
                    });
                });
            });
        });
    if st.capture.camera.mode == CameraMode::Orbit && !recording {
        ctx.request_repaint(); // the gizmo's orbit preview turns
    }
    if ctx.input(|i| i.key_pressed(egui::Key::Escape)) && r.response.contains_pointer() {
        st.capture.open = false;
    }
    r.response.rect.bottom()
}

/// The capture camera: the view's, a fixed angle or an orbit — the azimuth on a dial seen from
/// above, the elevation on a half dial seen from the side, the distance and the orbit's rate.
fn camera_controls(ui: &mut egui::Ui, c: &mut CameraSettings, window: &render::camera::OrbitCamera) {
    let before = c.mode;
    if let Some(m) = theme::segmented(
        ui,
        c.mode,
        &[
            (CameraMode::View, "the view", "The window's camera, cropped to the frame's aspect (the outlined frame over the view)."),
            (CameraMode::Fixed, "fixed angle", "A fixed angle and distance about the cell's centre (it stays with the box when you pan), shown as a camera in the view."),
            (CameraMode::Orbit, "orbit", "Round the cell's centre at a steady rate per second of video, from the fixed angle."),
        ],
    ) {
        c.mode = m;
        if before == CameraMode::View && m != CameraMode::View {
            c.from_view(window); // start from what is on screen
        }
    }
    if c.mode == CameraMode::View {
        return;
    }
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            // the window camera's direction from the cell's centre (the rig's centre)
            let eye = window.eye();
            azimuth_dial(ui, &mut c.azimuth_deg, eye.x.atan2(eye.z).to_degrees());
            theme::note(ui, format!("azimuth {:.0}°", c.azimuth_deg));
        });
        ui.vertical(|ui| {
            elevation_dial(ui, &mut c.elevation_deg);
            theme::note(ui, format!("elevation {:+.0}°", c.elevation_deg));
        });
    });
    ui.horizontal(|ui| {
        ui.label("distance");
        ui.add(egui::Slider::new(&mut c.distance, 1.0..=200.0).logarithmic(true).max_decimals(1).suffix(" Bohr"));
    });
    if c.mode == CameraMode::Orbit {
        ui.horizontal(|ui| {
            ui.label("orbit");
            ui.add(egui::Slider::new(&mut c.orbit_deg_per_s, -120.0..=120.0).max_decimals(0).suffix("° / s of video"));
        });
        if c.orbit_deg_per_s.abs() > 0.5 {
            theme::note(ui, format!("one turn per {:.0} s of video", 360.0 / c.orbit_deg_per_s.abs()));
        }
    }
    if ui.small_button("match the view").on_hover_text("Put the capture camera where the window's camera is (its direction and distance from the cell's centre)").clicked() {
        c.from_view(window);
    }
}

/// The azimuth seen from above: the scene at the centre, the capture camera on the ring (drag it
/// round), the view's camera as a faint line.
fn azimuth_dial(ui: &mut egui::Ui, az_deg: &mut f32, view_deg: f32) {
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(96.0), egui::Sense::click_and_drag());
    let p = ui.painter();
    let c = rect.center();
    let r = rect.width() * 0.4;
    p.circle_filled(c, r + 6.0, theme::APP);
    p.circle_stroke(c, r, Stroke::new(1.0, theme::LINE));
    for k in 0..8 {
        let a = (k as f32 * 45f32).to_radians();
        let d = Vec2::new(a.sin(), a.cos());
        p.line_segment([c + d * (r - 4.0), c + d * r], Stroke::new(1.0, theme::LINE));
    }
    let at = |deg: f32| {
        let a = deg.to_radians();
        c + Vec2::new(a.sin(), a.cos()) * r
    };
    p.line_segment([c, at(view_deg)], Stroke::new(1.0, theme::INK_FAINT));
    p.circle_filled(c, 3.5, theme::INK_DIM);
    p.line_segment([c, at(*az_deg)], Stroke::new(1.5, AMBER_GUIDE.gamma_multiply(0.7)));
    p.circle_filled(at(*az_deg), 6.0, AMBER_GUIDE);
    if let Some(pos) = resp.interact_pointer_pos().filter(|_| resp.dragged() || resp.clicked()) {
        let d = pos - c;
        if d.length() > 2.0 {
            *az_deg = camera::wrap_deg(d.x.atan2(d.y).to_degrees().round());
        }
    }
    resp.on_hover_text("Seen from above: drag the dot round the scene (the faint line is the view's camera)");
}

/// The elevation seen from the side: the target's horizontal plane through the centre, the camera
/// on the half ring (drag it up or down).
fn elevation_dial(ui: &mut egui::Ui, el_deg: &mut f32) {
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(70.0, 96.0), egui::Sense::click_and_drag());
    let p = ui.painter();
    let c = pos2(rect.left() + 10.0, rect.center().y);
    let r = 40.0;
    let arc: Vec<egui::Pos2> = (0..=32)
        .map(|k| {
            let a = (-90.0 + 180.0 * k as f32 / 32.0f32).to_radians();
            c + Vec2::new(a.cos(), -a.sin()) * r
        })
        .collect();
    p.add(egui::Shape::line(arc, Stroke::new(1.0, theme::LINE)));
    p.line_segment([c, c + Vec2::new(r + 6.0, 0.0)], Stroke::new(1.0, theme::INK_FAINT));
    p.circle_filled(c, 3.5, theme::INK_DIM);
    let a = el_deg.to_radians();
    let at = c + Vec2::new(a.cos(), -a.sin()) * r;
    p.line_segment([c, at], Stroke::new(1.5, AMBER_GUIDE.gamma_multiply(0.7)));
    p.circle_filled(at, 6.0, AMBER_GUIDE);
    if let Some(pos) = resp.interact_pointer_pos().filter(|_| resp.dragged() || resp.clicked()) {
        let d = pos - c;
        *el_deg = (-d.y).atan2(d.x.max(0.0)).to_degrees().round().clamp(-camera::MAX_ELEVATION, camera::MAX_ELEVATION);
    }
    resp.on_hover_text("Seen from the side: drag the dot up to look down on the scene, down to look up at it");
}

fn open_folder(dir: &std::path::Path) {
    #[cfg(windows)]
    let program = "explorer";
    #[cfg(target_os = "macos")]
    let program = "open";
    #[cfg(all(unix, not(target_os = "macos")))]
    let program = "xdg-open";
    std::process::Command::new(program).arg(dir).spawn().ok();
}
