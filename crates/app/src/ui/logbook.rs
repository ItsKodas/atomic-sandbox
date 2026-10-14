//! The session log behind the dock's Log tab, the toasts and the strip's counter: every message
//! the old side panel showed in place (UI errors, solver failures, refusals, box / dt / evaluation
//! warnings, dynamics and evaluation notes, sampling-pass summaries, model warnings, neutral
//! status lines) is recorded here when it first appears, newest last, with the session time.
//! The places that showed it before still show the current one (the status chip, Readouts ▸
//! Health, the Model tab); the log keeps the history.
use super::{box_warning, PanelInput};
use crate::sim::sampling_active;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Entries kept (the oldest drop first).
const KEEP: usize = 500;
/// A note or warning repeated within this many newest entries is not logged again (evaluation
/// notes repeat step after step).
const DEDUPE_WINDOW: usize = 12;
/// How long a status line or a refusal shows as a toast.
pub const TOAST_FOR: Duration = Duration::from_secs(6);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogKind {
    Error,
    Warning,
    Refused,
    Note,
    Pass,
    Status,
}

impl LogKind {
    pub fn label(self) -> &'static str {
        match self {
            LogKind::Error => "error",
            LogKind::Warning => "warning",
            LogKind::Refused => "refused",
            LogKind::Note => "note",
            LogKind::Pass => "pass",
            LogKind::Status => "status",
        }
    }

    pub fn colour(self) -> egui::Color32 {
        match self {
            LogKind::Error => super::RED,
            LogKind::Warning | LogKind::Refused => super::AMBER,
            LogKind::Note | LogKind::Pass | LogKind::Status => super::theme::INK_DIM,
        }
    }
}

/// The Log tab's filter chips.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogFilter {
    All,
    Errors,
    Refused,
    Warnings,
    Notes,
    Status,
}

impl LogFilter {
    pub fn admits(self, k: LogKind) -> bool {
        match self {
            LogFilter::All => true,
            LogFilter::Errors => k == LogKind::Error,
            LogFilter::Refused => k == LogKind::Refused,
            LogFilter::Warnings => k == LogKind::Warning,
            LogFilter::Notes => matches!(k, LogKind::Note | LogKind::Pass),
            LogFilter::Status => k == LogKind::Status,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LogEntry {
    /// Session time (since the viewer started).
    pub at: Duration,
    pub kind: LogKind,
    pub text: String,
}

/// What the recorder saw last, per source (a source logs again only when it changes).
#[derive(Default)]
struct Seen {
    status: String,
    status_seq: u64,
    /// The refusal's text and the snapshot's driver-command count: a repeated refusal of the same
    /// command (another Space during a pass, the same dt typed again) is a new refusal.
    refusal: Option<(String, u64)>,
    /// Evaluation notes and warnings already logged this run (each distinct text once per run: they
    /// repeat step after step), and the step the run was at (a smaller one is a new run).
    eval: std::collections::HashSet<String>,
    eval_step: u64,
    ui_error: Option<String>,
    solver_error: Option<String>,
    box_warning: Option<(bool, usize)>,
    dt_warning: Option<String>,
    notes_total: u64,
    pass: Option<String>,
    provenance: Option<String>,
    autoload: Option<String>,
}

pub struct LogBook {
    start: Instant,
    pub entries: VecDeque<LogEntry>,
    /// Errors and refusals logged since the Log tab was last open (the strip's counter).
    pub unread: usize,
    /// The newest status line or refusal and when it was logged (the toast).
    pub toast: Option<(Instant, LogKind, String)>,
    seen: Seen,
}

impl Default for LogBook {
    fn default() -> LogBook {
        LogBook { start: Instant::now(), entries: VecDeque::new(), unread: 0, toast: None, seen: Seen::default() }
    }
}

/// "h:mm:ss" of session time.
pub fn clock(d: Duration) -> String {
    let s = d.as_secs();
    format!("{}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

impl LogBook {
    pub fn push(&mut self, kind: LogKind, text: impl Into<String>) {
        let text = text.into();
        if matches!(kind, LogKind::Note | LogKind::Warning)
            && self.entries.iter().rev().take(DEDUPE_WINDOW).any(|e| e.kind == kind && e.text == text)
        {
            return;
        }
        if matches!(kind, LogKind::Error | LogKind::Refused) {
            self.unread += 1;
        }
        if matches!(kind, LogKind::Status | LogKind::Refused) {
            self.toast = Some((Instant::now(), kind, text.clone()));
        }
        self.entries.push_back(LogEntry { at: self.start.elapsed(), kind, text });
        while self.entries.len() > KEEP {
            self.entries.pop_front();
        }
    }

    pub fn count(&self, kind: LogKind) -> usize {
        self.entries.iter().filter(|e| e.kind == kind).count()
    }

    /// The toast to show now, if any.
    pub fn current_toast(&self) -> Option<(LogKind, &str)> {
        self.toast
            .as_ref()
            .filter(|(t, _, _)| t.elapsed() < TOAST_FOR)
            .map(|(_, k, s)| (*k, s.as_str()))
    }

    /// Logs whatever is new in this frame's inputs (`status` and `status_seq` = `UiState`'s: a
    /// status set again with the same text is a new message).
    pub fn record(&mut self, status: &str, status_seq: u64, input: &PanelInput<'_>) {
        if status != self.seen.status || status_seq != self.seen.status_seq {
            self.seen.status = status.to_string();
            self.seen.status_seq = status_seq;
            if !status.is_empty() {
                self.push(LogKind::Status, status);
            }
        }
        let ui_error = input.last_error.map(str::to_string);
        if ui_error != self.seen.ui_error {
            if let Some(e) = &ui_error {
                self.push(LogKind::Error, e.clone());
            }
            self.seen.ui_error = ui_error;
        }
        let Some(s) = input.snapshot else { return };
        let d = &s.dynamics;
        if d.last_error != self.seen.solver_error {
            if let Some(e) = &d.last_error {
                self.push(LogKind::Error, format!("paused by a solver failure: {e}"));
            }
            self.seen.solver_error = d.last_error.clone();
        }
        let refusal = d.last_rejection.clone().map(|r| (r, s.driver_commands));
        if refusal != self.seen.refusal {
            if let Some((r, _)) = &refusal {
                self.push(LogKind::Refused, format!("refused: {r}"));
            }
            self.seen.refusal = refusal;
        }
        // the box warning logs when it starts or names another nucleus, not on every 0.1 Bohr
        let bw = box_warning(&s.scene);
        let key = bw.as_ref().map(|(c, _, i)| (*c == super::RED, *i));
        if key != self.seen.box_warning {
            if let Some((_, text, _)) = bw {
                self.push(LogKind::Warning, text);
            }
            self.seen.box_warning = key;
        }
        if d.dt_warning != self.seen.dt_warning {
            if let Some(w) = &d.dt_warning {
                self.push(LogKind::Warning, w.clone());
            }
            self.seen.dt_warning = d.dt_warning.clone();
        }
        // a new worker restarts its note counter
        if d.notes_total < self.seen.notes_total {
            self.seen.notes_total = 0;
        }
        let fresh = ((d.notes_total - self.seen.notes_total) as usize).min(d.notes.len());
        for n in &d.notes[d.notes.len() - fresh..] {
            self.push(LogKind::Note, n.clone());
        }
        self.seen.notes_total = d.notes_total;
        if d.step < self.seen.eval_step {
            self.seen.eval.clear();
        }
        self.seen.eval_step = d.step;
        for (kind, text) in d.eval.warnings.iter().map(|w| (LogKind::Warning, w)).chain(d.eval.notes.iter().map(|n| (LogKind::Note, n))) {
            if self.seen.eval.insert(text.clone()) {
                self.push(kind, text.clone());
            }
        }
        let pass = d.sampling.as_ref().filter(|_| !sampling_active(d)).map(|p| {
            format!(
                "last pass {}: {}/{} jobs, skipped {}, failed {}",
                if p.cancelled { "cancelled" } else { "finished" },
                p.done,
                p.total,
                p.skipped,
                p.failed
            )
        });
        if pass != self.seen.pass {
            if let Some(p) = &pass {
                self.push(LogKind::Pass, p.clone());
            }
            self.seen.pass = pass;
        }
        let m = s.model.as_ref();
        let prov = m.and_then(|m| m.provenance_mismatch.clone());
        if prov != self.seen.provenance {
            if let Some(why) = &prov {
                self.push(LogKind::Warning, format!("{why}: every step falls back to DFT and teaches nothing — restore those settings, or forget the model and sample this scene"));
            }
            self.seen.provenance = prov;
        }
        let auto = m.and_then(|m| m.autoload_refused.clone());
        if auto != self.seen.autoload {
            if let Some(r) = &auto {
                self.push(LogKind::Warning, format!("this session's model was not restored: {r}"));
            }
            self.seen.autoload = auto;
        }
    }
}

