//! The two docks. The LEFT dock is what is going on (Activity, Readouts, Graph: stats), the RIGHT
//! one is what you configure (Model, Orbitals, Selection, Log) — user request 2026-10-19. Both are
//! built the same way: an icon strip (one button per tab, with the status dots: drift and box
//! health on Readouts, work running on Activity, trust on Model, the unread error / refusal count
//! on Log; hovering a button peeks its key numbers) and, when open, the tab's panel (resizable).
//! Clicking the open tab's button, or the chevron, collapses that dock to its strip. Each side has
//! its own open flag, selected tab and ONE width shared by its tabs (`UiState::left_*` / `dock_*`);
//! the panel sits at the window's edge and the strip next to the 3D view, mirrored on the left.
//! All of it is display state: none of it is an `Action`, so nothing here is recorded.
mod compare;
mod graph;
mod log;
pub mod model;
pub mod orbitals;
mod readouts;
mod selection;

#[cfg(test)]
pub use log::last_refusal;
pub use model::{model_file_buttons, offer, progress, teach_row};
#[cfg(test)]
pub use model::{AUDIT_OFF_NOTE, AUDIT_WHY};

use super::icons::{self, Icon};
use super::{box_warning, drift_colour, theme, trust, Action, DockSide, DockTab, EnergyUnits, PanelInput, UiState};
use crate::sim::Backend;
use egui::{Color32, Frame, Margin, RichText, Stroke, Ui, Vec2};

pub fn show(root: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    side(root, st, input, acts, DockSide::Left);
    side(root, st, input, acts, DockSide::Right);
    st.dock_w_set = None; // a profile's widths are applied for one frame
    drag_tabs(root.ctx(), st);
    if st.tab_shown(DockTab::Log) {
        st.log.unread = 0;
    }
}

/// One dock: its panel (when open) at the window's edge, then its strip next to the 3D view. The
/// `view` the chrome leaves (`PanelOutput::view`) is what remains between the two docks.
fn side(root: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>, side: DockSide) {
    let (panel, strip_panel) = match side {
        DockSide::Left => (egui::Panel::left("ldock-body"), egui::Panel::left("ldock-strip")),
        DockSide::Right => (egui::Panel::right("dock-body"), egui::Panel::right("dock-strip")),
    };
    if *st.dock_mut(side).0 {
        // one width for every tab of a side: a resize carries over when you switch tabs (user
        // request 2026-10-18; the Activity tab used to keep its own)
        let margin = match side {
            DockSide::Left => Margin { left: 8, right: 12, top: 8, bottom: 8 },
            DockSide::Right => Margin { left: 12, right: 8, top: 8, bottom: 8 },
        };
        let panel = match st.dock_w_set {
            Some(w) => panel.exact_size(w[side as usize]),
            None => panel.default_size(336.0), // 420 − 20 % (user request 2026-10-19)
        };
        let shown = panel
            .resizable(true)
            .size_range(280.0..=720.0)
            .frame(Frame::new().fill(theme::PANEL).inner_margin(margin))
            .show(root, |ui| body(ui, st, input, acts, side));
        st.dock_w[side as usize] = shown.response.rect.width();
    }
    strip_panel
        .exact_size(44.0)
        .resizable(false)
        .frame(Frame::new().fill(theme::PANEL).inner_margin(Margin::symmetric(5, 8)).stroke(Stroke::new(1.0, theme::LINE)))
        .show(root, |ui| strip(ui, st, input, side));
}

/// A grid spanning the dock's full width: each of its `cols` columns is at least an equal share,
/// and long values wrap at the dock's edge instead of a fixed column width.
pub(super) fn wide_grid(ui: &Ui, id: &str, cols: usize) -> egui::Grid {
    let w = ui.available_width();
    let gap = ui.spacing().item_spacing.x * (cols.saturating_sub(1)) as f32;
    let share = ((w - gap) / cols.max(1) as f32).max(40.0);
    egui::Grid::new(id).num_columns(cols).min_col_width(share).max_col_width(w.max(share))
}

fn icon(tab: DockTab) -> Icon {
    match tab {
        DockTab::Readouts => Icon::Readouts,
        DockTab::Graph => Icon::Graph,
        DockTab::Model => Icon::Model,
        DockTab::Activity => Icon::Activity,
        DockTab::Orbitals => Icon::Orbitals,
        DockTab::Selection => Icon::Selection,
        DockTab::Log => Icon::Log,
    }
}

/// The Readouts button's dot: the drift colour above 1e-4 Ha/ps, else the box warning's.
fn readouts_dot(input: &PanelInput<'_>) -> Option<Color32> {
    let s = input.snapshot?;
    s.dynamics
        .drift_per_ps
        .and_then(drift_colour)
        .or_else(|| box_warning(&s.scene).map(|w| w.0))
}

/// The Model button's dot: the trust colour (estimated solver only).
fn model_dot(st: &UiState, input: &PanelInput<'_>) -> Option<Color32> {
    (input.snapshot?.backend == Backend::Estimated).then(|| trust(input.model(), input.scene, st).colour)
}

/// The Activity button's dot: work is running (a pass, a solve, a refit, an audit).
fn activity_dot(input: &PanelInput<'_>) -> bool {
    input.sampling() || input.probe.is_some_and(|p| p.main.running || p.side.running || p.refit.is_some())
}

/// The key numbers a strip button shows on hover.
fn peek(st: &UiState, input: &PanelInput<'_>, tab: DockTab) -> String {
    let d = input.snapshot.map(|s| &s.dynamics);
    match tab {
        DockTab::Readouts => match d {
            Some(d) => format!(
                "Readouts\nT {} · E total {:+.6} Ha\ndrift {}",
                super::averaged_temperature_text(d, &st.temp_avg, 1, st.temp_units),
                d.total,
                d.drift_per_ps.map_or("—".into(), |x| format!("{x:+.2e} Ha/ps"))
            ),
            None => "Readouts\nno solution yet".into(),
        },
        DockTab::Graph => match d {
            Some(d) => format!("Graph\nE total and T: {} points", d.history.len()),
            None => "Graph".into(),
        },
        DockTab::Model => match input.model() {
            Some(m) => format!("Model\ntrusted: {} · {} samples · epoch {}", trust(Some(m), input.scene, st).text, m.samples, m.epoch),
            None => "Model (estimated solver)\nnot running".into(),
        },
        DockTab::Activity => {
            let now = st.activity.headline.peek().as_ref().map_or("idle".to_string(), |h| h.text.clone());
            let counters = match input.model() {
                Some(m) => format!("{} samples · {} teacher solves for this model · {} teaching solves this session", m.samples, m.solves.total(), st.activity.session_solves),
                None => format!("{} teaching solves this session", st.activity.session_solves),
            };
            let newest = st.activity.rows.back().map_or(String::new(), |r| {
                format!("
newest: {} → {}", super::activity::log::rule_cell(&r.event), super::activity::words::outcome_label(r.event.outcome))
            });
            format!("Activity
{now}
{counters}{newest}")
        }
        DockTab::Orbitals => match input.snapshot.and_then(|s| s.dynamics.orbitals.as_ref()) {
            Some(o) if st.show != super::ShowMode::Density => {
                let l = orbitals::Levels::of(&o.rows);
                format!("Orbitals · {}\n{}", st.show.title(), orbitals::levels_line(&l).unwrap_or_else(|| format!("{} orbitals", o.rows.len())))
            }
            _ => format!("Orbitals\nshowing: {} (pick a mode to see the Kohn–Sham orbitals)", st.show.title()),
        },
        DockTab::Selection => match input.selection() {
            Some((i, n)) => format!("Selection\n{} #{i}", render::elements::symbol(n.z)),
            None => "Selection\nnothing selected".into(),
        },
        DockTab::Log => {
            let newest = st.log.entries.back().map_or(String::new(), |e| format!("\n{}: {}", e.kind.label(), e.text));
            format!("Log\n{} entries, {} unread errors or refusals{newest}", st.log.entries.len(), st.log.unread)
        }
    }
}

fn strip(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, side: DockSide) {
    let mut geom = Vec::new();
    ui.vertical_centered(|ui| {
        ui.spacing_mut().item_spacing.y = 4.0;
        for tab in st.tabs(side).to_vec() {
            let selected = st.tab_shown(tab);
            let tip = peek(st, input, tab);
            // a click opens the tab; a drag moves it (up or down, or onto the other strip)
            let r = icons::strip_button(ui, icon(tab), selected).interact(egui::Sense::click_and_drag());
            geom.push((tab, r.rect));
            if r.drag_started() {
                st.drag_tab = Some(tab);
            }
            r.context_menu(|ui| {
                let other = match st.side_of(tab) {
                    DockSide::Left => DockSide::Right,
                    DockSide::Right => DockSide::Left,
                };
                let name = if other == DockSide::Left { "Move to the left dock" } else { "Move to the right dock" };
                if ui.button(name).clicked() {
                    let n = st.tabs(other).len();
                    st.move_tab(tab, other, n);
                    ui.close();
                }
                if ui.button("Reset the tab layout").clicked() {
                    st.reset_layout();
                    ui.close();
                }
            });
            super::probe(&format!("dock-tab-{tab:?}"), &r);
            let p = ui.painter();
            let dot = match tab {
                DockTab::Readouts => readouts_dot(input),
                DockTab::Model => model_dot(st, input),
                DockTab::Activity => activity_dot(input).then_some(theme::ACCENT),
                _ => None,
            };
            if let Some(c) = dot {
                let at = r.rect.right_top() + Vec2::new(-8.0, 8.0);
                // the Activity dot pulses while work runs
                let pulse = if tab == DockTab::Activity { 0.85 + 0.15 * (ui.input(|i| i.time) * 7.0).sin() as f32 } else { 1.0 };
                p.circle_filled(at, 4.5, theme::PANEL);
                p.circle_filled(at, 3.5 * pulse, c);
            }
            if tab == DockTab::Log && st.log.unread > 0 {
                let at = r.rect.right_top() + Vec2::new(-8.0, 9.0);
                p.circle_filled(at, 7.5, super::RED);
                p.text(at, egui::Align2::CENTER_CENTER, st.log.unread.min(99).to_string(), egui::FontId::monospace(9.0), Color32::WHITE);
            }
            let hint = if selected { "\n(click to collapse the dock; drag to move the tab, right-click for more)" } else { "\n(drag to move the tab, right-click for more)" };
            if r.on_hover_text(format!("{tip}{hint}")).clicked() {
                if selected {
                    *st.dock_mut(side).0 = false;
                } else {
                    st.open_tab(tab);
                }
            }
        }
        ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
            let open = *st.dock_mut(side).0;
            // the chevron points the way the panel goes: toward the window's edge when collapsing
            let toward_right = (side == DockSide::Right) == open;
            let (i, tip) = (if toward_right { Icon::ChevronRight } else { Icon::ChevronLeft }, if open { "Collapse the dock" } else { "Open the dock" });
            let r = icons::strip_button(ui, i, false);
            super::probe(&format!("dock-chevron-{side:?}"), &r);
            if r.on_hover_text(tip).clicked() {
                *st.dock_mut(side).0 = !open;
            }
        });
    });
    st.strip_geom[side as usize] = Some((ui.max_rect(), geom));
}

/// Where a dragged tab would land: the strip under the pointer (a little wider than drawn) and
/// the index among its other tabs, by the buttons' centres.
fn drop_target(st: &UiState, tab: DockTab, at: egui::Pos2) -> Option<(DockSide, usize, f32, egui::Rect)> {
    for side in [DockSide::Left, DockSide::Right] {
        let Some((rect, buttons)) = &st.strip_geom[side as usize] else { continue };
        if !rect.expand2(Vec2::new(24.0, 0.0)).contains(at) {
            continue;
        }
        let others: Vec<&(DockTab, egui::Rect)> = buttons.iter().filter(|(t, _)| *t != tab).collect();
        let index = others.iter().filter(|(_, r)| r.center().y < at.y).count();
        let y = match (index.checked_sub(1).and_then(|i| others.get(i)), others.get(index)) {
            (_, Some((_, r))) => r.top() - 2.0,
            (Some((_, r)), None) => r.bottom() + 2.0,
            (None, None) => rect.top() + 6.0,
        };
        return Some((side, index, y, *rect));
    }
    None
}

/// The tab being dragged: its icon under the pointer, the line where it would land, and the move
/// on release (nowhere near a strip: nothing moves).
fn drag_tabs(ctx: &egui::Context, st: &mut UiState) {
    let Some(tab) = st.drag_tab else { return };
    // released outside the window: the drag is over without a drop
    if !ctx.input(|i| i.pointer.any_down() || i.pointer.any_released()) {
        st.drag_tab = None;
        return;
    }
    let Some(at) = ctx.pointer_interact_pos() else { return };
    let target = drop_target(st, tab, at);
    let p = ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("dock-drag")));
    if let Some((_, _, y, rect)) = target {
        p.line_segment([egui::pos2(rect.left() + 4.0, y), egui::pos2(rect.right() - 4.0, y)], Stroke::new(2.0, theme::ACCENT));
    }
    let r = egui::Rect::from_center_size(at, Vec2::splat(30.0));
    p.rect_filled(r, 6.0, theme::ACCENT_BG.gamma_multiply(0.9));
    icons::paint(&p, r.shrink(7.0), icon(tab), theme::ACCENT_INK);
    ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
    if ctx.input(|i| i.pointer.any_released()) {
        st.drag_tab = None;
        if let Some((side, index, _, _)) = target {
            st.move_tab(tab, side, index);
        }
    }
}

fn body(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>, side: DockSide) {
    let tab = *st.dock_mut(side).1;
    ui.horizontal(|ui| {
        ui.label(RichText::new(tab.title()).strong().size(14.0));
        if tab == DockTab::Readouts {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if let Some(u) = theme::segmented(
                    ui,
                    st.energy_units,
                    &[(EnergyUnits::Both, "both", ""), (EnergyUnits::Ev, "eV", ""), (EnergyUnits::Ha, "Ha", "")],
                ) {
                    st.energy_units = u;
                }
            });
        }
    });
    ui.add(egui::Separator::default().spacing(6.0));
    egui::ScrollArea::vertical()
        .id_salt(("dock", tab as u8))
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 6.0;
            match tab {
                DockTab::Readouts => readouts::show(ui, st, input, acts),
                DockTab::Graph => graph::show(ui, st, input),
                DockTab::Model => model::show(ui, st, input, acts),
                DockTab::Activity => super::activity::show(ui, st, input, acts),
                DockTab::Orbitals => orbitals::show(ui, st, input, acts),
                DockTab::Selection => selection::show(ui, st, input, acts),
                DockTab::Log => log::show(ui, st, input),
            }
        });
}

/// A titled group in a tab, marked with its search anchor.
pub(super) fn group<R>(ui: &mut Ui, st: &mut UiState, anchor: &str, add: impl FnOnce(&mut Ui, &mut UiState) -> R) -> R {
    let r = ui.scope(|ui| add(ui, st));
    super::search::mark(ui, st, anchor, r.response.rect);
    r.inner
}
