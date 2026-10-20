//! The live trust checklist (activity spec §3.3.4, §5.3): the trust rule's conditions in the order
//! it tests them, each ✓ or ✗ with its number against its limit, in plain words. The first ✗ is
//! the reason the model is refused ("this is why"); when the box is what blocks trust it comes
//! first. The checklist and the refusal come from one predicate per condition (plan review M2), so
//! they agree by construction.
use super::words::{self, AuditSeen};
use crate::ui::{theme, PanelInput, UiState, AMBER, GREEN};
use egui::{RichText, Ui};

#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    pub met: bool,
    /// The refusal's condition ("this is why").
    pub why: bool,
    pub text: String,
    pub hover: String,
}

/// What the classes mean (plan review N8: two separate H₂ molecules are "two-atom").
pub const CLASS_HELP: &str = "two-atom: every atom has at most one neighbour within the model's reach (two separate H₂ molecules count too); three or more: some atom has two or more";

/// The lines of the checklist for a fitted model.
pub fn lines(m: &crate::sim::ModelStatus, scene: &crate::scene::Scene, st: &UiState) -> Vec<Line> {
    let mut v = Vec::new();
    let boxed = crate::sim::box_blocks_trust(m);
    if let Some((b, n)) = boxed {
        v.push(Line {
            met: false,
            why: true,
            // b counts the box-affected samples in the fit, n the samples held (plan review N6)
            text: format!("Samples clear of the box walls — {b} in the fit touch them ({n} held). {}", crate::ui::box_advice(scene, st)),
            hover: format!("the fit's own check failed: {}", m.untrusted_reason.as_deref().unwrap_or("?")),
        });
    }
    let first = m.trust_checks.iter().position(|c| !c.met);
    let seen = AuditSeen::of(m);
    for (k, c) in m.trust_checks.iter().enumerate() {
        v.push(Line {
            met: c.met,
            why: boxed.is_none() && Some(k) == first,
            text: words::trust_line(c, seen, m.audit_every),
            hover: format!("value {:.3e}, limit {:.3e}", c.value, c.limit),
        });
    }
    if boxed.is_none() && first.is_none() && m.fitted {
        // box-affected samples below the trust floor are fitted, and a model can still pass with
        // them: the line says what is true (activity final review M3)
        let in_fit = m.box_affected.saturating_sub(m.box_excluded);
        let mut text = if in_fit == 0 {
            "Samples clear of the box walls".to_string()
        } else {
            format!("Box walls not blocking trust — {in_fit} samples in the fit touch them ({} held)", m.samples)
        };
        // fix (d), 2026-10-18: the box-affected samples left out of the fit are not training
        // progress — say how many the walls cost, every time
        if m.box_excluded > 0 {
            text = format!("{text}; {} box-affected samples are left out of the fit (they taught nothing: a bigger cell would)", m.box_excluded);
        }
        v.push(Line { met: true, why: false, text, hover: "box-affected samples are not keeping the model untrusted".into() });
    }
    // the strict policy at this geometry (fix (a), 2026-10-18): the line that says whether the
    // checklist above means the model answers *here*; "this is why" when everything above is met
    if let Some(r) = &m.readiness {
        v.push(Line {
            met: r.answers,
            why: !r.answers && boxed.is_none() && first.is_none(),
            text: if r.answers {
                "Strict mode answers at this geometry".to_string()
            } else {
                format!("Strict mode answers at this geometry — no: {}{}", r.text, if m.draft { " (draft mode is answering anyway)" } else { "" })
            },
            hover: format!("the fallback policy's verdict here, draft mode off: rule {:?}{}", r.rule, r.atom.map_or(String::new(), |a| format!(", atom {a}"))),
        });
        if let Some(u) = &r.unlearned {
            v.push(Line { met: false, why: r.answers && boxed.is_none() && first.is_none(), text: format!("This scene's own sampling pass is in the fit — no: {u}"), hover: "the last sampling pass of these species: what its samples became in the fit that stands".into() });
        }
    }
    for l in &m.class_loo {
        v.push(Line {
            met: l.trusted,
            why: false,
            text: words::class_line(l),
            hover: format!(
                "{} samples, leave-one-out force RMS {:.2}× / max {:.2}× the bound, energy {:.2}× the tier; {CLASS_HELP}",
                l.samples, l.force_rms, l.force_max, l.energy_max
            ),
        });
    }
    v
}

/// ✓ or ✗, drawn (the UI fonts have neither glyph).
fn mark(ui: &mut Ui, met: bool) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 15.0), egui::Sense::hover());
    let c = rect.center();
    let p = ui.painter();
    if met {
        p.add(egui::Shape::line(vec![c + egui::vec2(-4.5, 0.5), c + egui::vec2(-1.5, 3.5), c + egui::vec2(4.5, -3.5)], egui::Stroke::new(1.8, GREEN)));
    } else {
        let s = egui::Stroke::new(1.8, AMBER);
        p.line_segment([c + egui::vec2(-3.5, -3.5), c + egui::vec2(3.5, 3.5)], s);
        p.line_segment([c + egui::vec2(-3.5, 3.5), c + egui::vec2(3.5, -3.5)], s);
    }
}

/// The tab's trust group: ✓ / ✗ lines, "this is why" as an amber pill on the refusal's line, and
/// the stuck variant when a trusted model keeps falling back.
pub fn show(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<crate::ui::actions::Action>) {
    let Some(m) = input.model().filter(|m| m.fitted) else {
        // samples held but the fit did not form: why, and the Retry (the Model tab shows the same)
        match input.model().filter(|m| m.no_model.is_some()) {
            Some(m) => crate::ui::dock::model::no_model_row(ui, m, input.sampling(), acts),
            None => {
                theme::note(ui, "no model yet — the checklist appears once a sampling pass has fitted one");
            }
        }
        return;
    };
    let ls = lines(m, input.scene, st);
    for l in &ls {
        ui.horizontal_wrapped(|ui| {
            mark(ui, l.met);
            if l.why {
                theme::pill(ui, "this is why", AMBER);
            }
            ui.add(egui::Label::new(RichText::new(&l.text).size(12.0).color(theme::INK)).wrap()).on_hover_text(&l.hover);
        });
    }
    crate::ui::dock::model::teach_row(ui, m, input.sampling(), acts);
    if m.trusted {
        if let (Some(info), Some(snap)) = (&m.stuck_info, input.snapshot) {
            ui.horizontal_wrapped(|ui| {
                theme::light(ui, AMBER);
                let r = ui.add(egui::Label::new(RichText::new(words::stuck_sentence(info, &snap.dynamics.scene)).size(12.0)).wrap());
                if let Some(old) = &m.stuck {
                    r.on_hover_text(old);
                }
            });
        }
    }
}
