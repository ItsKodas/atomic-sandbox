//! The workspace's panels (docking spec D-2: "dock tab" now means "panel"): what each tab draws
//! (`body`: its title row, then its own scroll area round the per-panel `show`), its icon, its
//! status (`status`: drift and box health on Readouts, work running on Activity, trust on Model,
//! the unread error / refusal count on Log — the old icon strip's dots, spec D-4) and the key
//! numbers its tab peeks on hover (`peek`). Where a panel sits — which leaf, which split, which
//! window — is the workspace tree's (`super::workspace::tree`), which draws the tabs and frames these
//! bodies. All of it is display state: none of it is an `Action`, so nothing here is recorded.
mod compare;
mod log;
pub mod model;
pub mod orbitals;
pub mod readouts;
pub mod selection;
pub mod view;

#[cfg(test)]
pub use log::last_refusal;
pub use model::{model_file_buttons, offer, progress, teach_row};
#[cfg(test)]
pub use model::{AUDIT_OFF_NOTE, AUDIT_WHY};

use super::icons::Icon;
use super::{box_warning, drift_colour, theme, trust, Action, DockTab, EnergyUnits, PanelInput, UiState};
use crate::sim::Backend;
use egui::{Color32, RichText, Ui};

/// A grid spanning the dock's full width: each of its `cols` columns is at least an equal share,
/// and long values wrap at the dock's edge instead of a fixed column width.
pub(super) fn wide_grid(ui: &Ui, id: &str, cols: usize) -> egui::Grid {
    let w = ui.available_width();
    let gap = ui.spacing().item_spacing.x * (cols.saturating_sub(1)) as f32;
    let share = ((w - gap) / cols.max(1) as f32).max(40.0);
    // a column may grow to what the others leave at their least (it used to be the whole width,
    // so a long value ran past the panel's right edge instead of wrapping)
    let most = (w - gap - share * (cols.saturating_sub(1)) as f32).max(share);
    egui::Grid::new(id).num_columns(cols).min_col_width(share).max_col_width(most)
}

/// The panel's icon (its tab, docking spec D-4).
pub(super) fn icon(tab: DockTab) -> Icon {
    match tab {
        DockTab::Readouts => Icon::Readouts,
        DockTab::Model => Icon::Model,
        DockTab::Activity => Icon::Activity,
        DockTab::Orbitals => Icon::Orbitals,
        DockTab::Selection => Icon::Selection,
        DockTab::Log => Icon::Log,
        DockTab::Knowledge => Icon::Knowledge,
        DockTab::View => Icon::View,
    }
}

/// The Readouts tab's dot: the drift colour above 1e-4 Ha/ps, else the box warning's.
fn readouts_dot(input: &PanelInput<'_>) -> Option<Color32> {
    let s = input.snapshot?;
    s.dynamics
        .drift_per_ps
        .and_then(drift_colour)
        .or_else(|| box_warning(&s.scene).map(|w| w.0))
}

/// The Model tab's dot: the trust colour (estimated solver only).
fn model_dot(st: &UiState, input: &PanelInput<'_>) -> Option<Color32> {
    (input.snapshot?.backend == Backend::Estimated).then(|| trust(input.model(), input.scene, st).colour)
}

/// The Activity tab's dot: work is running (a pass, a solve, a refit, an audit).
fn activity_dot(input: &PanelInput<'_>) -> bool {
    input.sampling() || input.probe.is_some_and(|p| p.main.running || p.side.running || p.refit.is_some())
}

/// Whether `tab` can carry a status dot ([`status`]): its tab keeps the dot's room always, so its
/// width holds still when the dot comes and goes.
pub(super) fn can_flag(tab: DockTab) -> bool {
    matches!(tab, DockTab::Readouts | DockTab::Model | DockTab::Activity | DockTab::Log)
}

/// What a panel's tab shows after its name (docking spec D-4, the old strip's dots): the dot's
/// colour, and the text in it when it counts something (the Log's unread errors and refusals).
/// `None`: nothing to flag. Only the [`can_flag`] tabs have one.
pub(super) fn status(tab: DockTab, st: &UiState, input: &PanelInput<'_>) -> Option<(Color32, Option<String>)> {
    match tab {
        DockTab::Readouts => readouts_dot(input).map(|c| (c, None)),
        DockTab::Model => model_dot(st, input).map(|c| (c, None)),
        DockTab::Activity => activity_dot(input).then_some((theme::ACCENT, None)),
        DockTab::Log => (st.log.unread > 0).then(|| (super::RED, Some(st.log.unread.min(99).to_string()))),
        _ => None,
    }
}

/// The key numbers a panel's tab shows on hover.
pub(super) fn peek(st: &UiState, input: &PanelInput<'_>, tab: DockTab) -> String {
    let d = input.snapshot.map(|s| &s.dynamics);
    match tab {
        DockTab::Readouts => match d {
            Some(d) => format!(
                "Readouts\nT {} · E total {:+.6} Ha\ndrift {}\nE total and T: {} points",
                super::averaged_temperature_text(d, &st.temp_avg, 1, st.temp_units),
                d.total,
                d.drift_per_ps.map_or("—".into(), |x| format!("{x:+.2e} Ha/ps")),
                d.history.len()
            ),
            None => "Readouts\nno solution yet".into(),
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
        DockTab::Knowledge => super::knowledge::peek(st),
        DockTab::View => "View
what the 3D view shows and how it looks".into(),
        DockTab::Log => {
            let newest = st.log.entries.back().map_or(String::new(), |e| format!("\n{}: {}", e.kind.label(), e.text));
            format!("Log\n{} entries, {} unread errors or refusals{newest}", st.log.entries.len(), st.log.unread)
        }
    }
}

/// A panel's body inside its tile: its title row (Readouts: with the energy units), a separator,
/// and its contents in a scroll area of their own.
pub(super) fn body(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>, tab: DockTab) {
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
    // the body's right edge: the tile's margin is the same on both sides (`workspace::tree`'s
    // tab body), and while the vertical bar shows it takes its own width instead of floating over
    // the content's right edge (user report 2026-11-14: cards and right-aligned words touched it)
    let scroll = &mut ui.spacing_mut().scroll;
    scroll.floating_allocated_width = scroll.floating_allocated_width.max(scroll.bar_width + 2.0);
    egui::ScrollArea::vertical()
        .id_salt(("dock", tab as u8))
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 6.0;
            // the body's visible part and its content rect (the tests' margin checks)
            super::probe_rect(&format!("dock-clip-{}", tab.name()), ui.clip_rect());
            super::probe_rect(&format!("dock-content-{}", tab.name()), ui.max_rect());
            match tab {
                DockTab::Readouts => readouts::show(ui, st, input, acts),
                DockTab::Model => model::show(ui, st, input, acts),
                DockTab::Activity => super::activity::show(ui, st, input, acts),
                DockTab::Orbitals => orbitals::show(ui, st, input, acts),
                DockTab::Selection => selection::show(ui, st, input, acts),
                DockTab::Log => log::show(ui, st, input),
                DockTab::Knowledge => super::knowledge::tab(ui, st, input, acts),
                DockTab::View => view::show(ui, st, acts),
            }
        });
}

/// A titled group in a tab, marked with its search anchor.
pub(super) fn group<R>(ui: &mut Ui, st: &mut UiState, anchor: &str, add: impl FnOnce(&mut Ui, &mut UiState) -> R) -> R {
    let r = ui.scope(|ui| add(ui, st));
    super::search::mark(ui, st, anchor, r.response.rect);
    r.inner
}
