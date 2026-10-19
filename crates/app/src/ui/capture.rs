//! The capture buttons at the top right of the 3D view (`crate::capture`): Screenshot, Record /
//! Stop and the quick settings popover (hide the UI, resolution, record mode, fs per frame). While
//! recording, a red dot with the frames, the simulated time covered and the video's length; after
//! it, the finish's state and the last note. Screenshot and Record go through [`Action`]; the
//! settings are display state written to `UiState::capture` directly.
use super::{icons, theme, Action, UiState, RED};
use crate::capture::{fs_words, RecordMode, Resolution};
use egui::{pos2, Align2, Id, Order, Rect, RichText, Stroke, Vec2};
use std::time::Duration;

/// How long the last note stays under the buttons.
const NOTE_FOR: Duration = Duration::from_secs(8);

/// Draws the buttons (top right of `view`) and returns the bottom of what they took, for the cards
/// stacked under them (the atom card).
pub fn show(ctx: &egui::Context, view: Rect, st: &mut UiState, acts: &mut Vec<Action>) -> f32 {
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
                            RecordMode::SimTime => format!("{} · {} frames · {:.1} s video", fs_words(r.sim_fs), r.frames, r.playback_s),
                            RecordMode::AsSeen => format!("{} frames · {:.1} s", r.frames, r.playback_s),
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
                if let Some(r) = &cv.recording {
                    for n in &r.notes {
                        ui.label(RichText::new(n).size(11.0).color(super::AMBER));
                    }
                }
                if let Some(f) = &cv.finishing {
                    ui.label(RichText::new(f).size(11.0).color(theme::INK_DIM));
                    ui.ctx().request_repaint_after(Duration::from_millis(200));
                } else if let Some((_, text, err)) = cv.last.as_ref().filter(|l| l.0.elapsed() < NOTE_FOR) {
                    ui.set_max_width(360.0);
                    ui.add(egui::Label::new(RichText::new(text).size(11.0).color(if *err { RED } else { theme::INK_DIM })).wrap());
                    ui.ctx().request_repaint_after(Duration::from_millis(500));
                }
            });
            let rect = ui.min_rect();
            super::search::mark(ui, st, "v-capture", rect);
        });
    let rect = area.response.rect;
    let mut bottom = rect.bottom();
    if st.capture.open {
        bottom = settings(ctx, rect, st, &cv).max(bottom);
    }
    bottom
}

/// The quick settings popover under the buttons; returns its bottom.
fn settings(ctx: &egui::Context, anchor: Rect, st: &mut UiState, cv: &crate::capture::View) -> f32 {
    let recording = cv.recording.is_some();
    let r = egui::Area::new(Id::new("capture-settings"))
        .pivot(Align2::RIGHT_TOP)
        .fixed_pos(pos2(anchor.right(), anchor.bottom() + 6.0))
        .order(Order::Foreground)
        .fade_in(false)
        .show(ctx, |ui| {
            theme::card().fill(theme::PANEL).stroke(Stroke::new(1.0, theme::LINE)).show(ui, |ui| {
                ui.set_width(300.0);
                let s = &mut st.capture;
                theme::section(ui, "Capture");
                let can_show_ui = cv.surface_copyable;
                ui.add_enabled_ui(can_show_ui, |ui| {
                    ui.checkbox(&mut s.hide_ui, "Hide the UI (the 3D view only)");
                });
                if !can_show_ui {
                    theme::note(ui, "this window's surface cannot be copied: captures are the 3D view only");
                }
                ui.add_enabled_ui(s.hide_ui || !can_show_ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Resolution");
                        for res in Resolution::ALL {
                            if ui.add(egui::Button::selectable(s.resolution == res, res.label())).clicked() {
                                s.resolution = res;
                            }
                        }
                    });
                });
                if !s.hide_ui && can_show_ui {
                    theme::note(ui, "with the UI shown the frame is the window, at its size");
                }
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
                if recording {
                    theme::note(ui, "changes apply to the next recording");
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
    // a click outside the popover and its buttons closes it
    let clicked_elsewhere = ctx.input(|i| i.pointer.any_click())
        && ctx.pointer_interact_pos().is_some_and(|p| !r.response.rect.contains(p) && !anchor.contains(p));
    if clicked_elsewhere {
        st.capture.open = false;
    }
    r.response.rect.bottom()
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
