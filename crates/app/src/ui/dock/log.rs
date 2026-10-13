//! The Log tab: everything the chrome reported this session, newest first, with filters
//! (`ui::logbook`). Above it, the driver's current refusal stays in place for as long as the driver
//! holds it (`DynSnapshot::last_rejection`, cleared by the next accepted command), as the old panel
//! showed it; the toast only flags it for 6 s.
use super::super::logbook::{clock, LogFilter, LogKind};
use super::super::{theme, PanelInput, UiState, AMBER};
use egui::{RichText, Ui};

/// The "last refusal" line: the refusal the driver currently holds, in amber (`None` without one).
pub fn last_refusal(input: &PanelInput<'_>) -> Option<String> {
    input
        .snapshot
        .and_then(|s| s.dynamics.last_rejection.as_ref())
        .map(|r| format!("last refusal: refused: {r}"))
}

pub fn show(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>) {
    if let Some(line) = last_refusal(input) {
        ui.label(RichText::new(line).color(AMBER))
            .on_hover_text("the command the driver refused last; it stays until the next accepted command (also in the list below)");
    }
    super::group(ui, st, "d-log", |ui, st| {
        ui.horizontal_wrapped(|ui| {
            let errors = st.log.count(LogKind::Error);
            let refused = st.log.count(LogKind::Refused);
            let warnings = st.log.count(LogKind::Warning);
            for (f, label) in [
                (LogFilter::All, "All".to_string()),
                (LogFilter::Errors, format!("Errors {errors}")),
                (LogFilter::Refused, format!("Refused {refused}")),
                (LogFilter::Warnings, format!("Warnings {warnings}")),
                (LogFilter::Notes, "Notes".into()),
                (LogFilter::Status, "Status".into()),
            ] {
                if ui.selectable_label(st.log_filter == f, label).clicked() {
                    st.log_filter = f;
                }
            }
        });
        ui.separator();
        let mut any = false;
        for e in st.log.entries.iter().rev().filter(|e| st.log_filter.admits(e.kind)) {
            any = true;
            ui.horizontal_top(|ui| {
                ui.label(RichText::new(clock(e.at)).monospace().size(10.5).color(theme::INK_FAINT));
                ui.add_sized(
                    [52.0, 14.0],
                    egui::Label::new(RichText::new(e.kind.label().to_uppercase()).monospace().size(9.5).color(e.kind.colour())),
                );
                let text = RichText::new(&e.text).size(11.5);
                let text = match e.kind {
                    LogKind::Error => text.color(super::super::RED),
                    LogKind::Note | LogKind::Pass | LogKind::Status => text.color(theme::INK_DIM),
                    _ => text,
                };
                ui.add(egui::Label::new(text).wrap());
            });
        }
        if !any {
            theme::note(ui, "nothing logged yet");
        }
        ui.add_space(6.0);
        theme::note(ui, "UI errors (invalid grid, unreadable file), solver failures, refusals, box warnings, evaluation warnings and notes, sampling summaries and status messages all collect here, newest first (session time). The newest error also turns the status chip red until the next edit.");
    });
}
