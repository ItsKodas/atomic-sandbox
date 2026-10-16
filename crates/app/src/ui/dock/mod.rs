//! The right dock: what you read. An icon strip (one button per tab, with the status dots: drift
//! and box health on Readouts, trust on Model, the unread error / refusal count on Log; hovering
//! a button peeks its key numbers) and, when open, the tab's panel (resizable). Clicking the
//! open tab's button, or the chevron, collapses the dock to the strip.
mod graph;
mod log;
mod model;
pub mod orbitals;
mod readouts;
mod selection;

#[cfg(test)]
pub use log::last_refusal;
pub use model::{model_file_buttons, offer, progress};

use super::icons::{self, Icon};
use super::{box_warning, drift_colour, theme, trust, Action, DockTab, EnergyUnits, PanelInput, UiState};
use crate::sim::Backend;
use egui::{Color32, Frame, Margin, RichText, Stroke, Ui, Vec2};

pub fn show(root: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    if st.dock_open {
        // the Activity tab keeps its own width: its log table wants more room than the others
        let activity = st.dock_tab == DockTab::Activity;
        egui::Panel::right(if activity { "dock-body-activity" } else { "dock-body" })
            .resizable(true)
            .default_size(if activity { 480.0 } else { 360.0 })
            .size_range(280.0..=720.0)
            .frame(Frame::new().fill(theme::PANEL).inner_margin(Margin { left: 12, right: 8, top: 8, bottom: 8 }))
            .show(root, |ui| body(ui, st, input, acts));
    }
    egui::Panel::right("dock-strip")
        .exact_size(44.0)
        .resizable(false)
        .frame(Frame::new().fill(theme::PANEL).inner_margin(Margin::symmetric(5, 8)).stroke(Stroke::new(1.0, theme::LINE)))
        .show(root, |ui| strip(ui, st, input));
    if st.dock_open && st.dock_tab == DockTab::Log {
        st.log.unread = 0;
    }
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
fn model_dot(input: &PanelInput<'_>) -> Option<Color32> {
    (input.snapshot?.backend == Backend::Estimated).then(|| trust(input.model()).colour)
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
                super::temperature_text(d, 1, st.temp_units),
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
            Some(m) => format!("Model\ntrusted: {} · {} samples · epoch {}", trust(Some(m)).text, m.samples, m.epoch),
            None => "Model (estimated solver)\nnot running".into(),
        },
        DockTab::Activity => {
            let now = st.activity.headline.peek().as_ref().map_or("idle".to_string(), |h| h.text.clone());
            format!("Activity
{now}
{} teaching solves this session", st.activity.session_solves)
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

fn strip(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>) {
    ui.vertical_centered(|ui| {
        ui.spacing_mut().item_spacing.y = 4.0;
        for tab in DockTab::ALL {
            let selected = st.dock_open && st.dock_tab == tab;
            let tip = peek(st, input, tab);
            let r = icons::strip_button(ui, icon(tab), selected);
            let p = ui.painter();
            let dot = match tab {
                DockTab::Readouts => readouts_dot(input),
                DockTab::Model => model_dot(input),
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
            let hint = if selected { "\n(click to collapse the dock)" } else { "" };
            if r.on_hover_text(format!("{tip}{hint}")).clicked() {
                if selected {
                    st.dock_open = false;
                } else {
                    st.dock_open = true;
                    st.dock_tab = tab;
                }
            }
        }
        ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
            let (i, tip) = if st.dock_open { (Icon::ChevronRight, "Collapse the dock") } else { (Icon::ChevronLeft, "Open the dock") };
            if icons::strip_button(ui, i, false).on_hover_text(tip).clicked() {
                st.dock_open = !st.dock_open;
            }
        });
    });
}

fn body(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    let tab = st.dock_tab;
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
                DockTab::Readouts => readouts::show(ui, st, input),
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
