//! The Model tab (the estimated solver, spec §6; rows 10–45 of the mapping; restructured
//! 2026-11-14 from the user's approved mockup — the old one was "information overload and it hides
//! all the helpful training stuff away below"). Top to bottom:
//!
//! 1. **The verdict card** ([`verdict`]): one headline — "Model answering · trusted here",
//!    "Falling back to DFT here", "No model yet", "Model only · guessing here" … — derived from the
//!    trust view, the strict policy's verdict here, draft / Model only and the step's source; one
//!    short sentence naming the reason (an unlearned pair is named); the long explanation (the old
//!    "trusted" and "strict here" paragraphs, the fit's own words) on the "why?" hover; the "N % DFT"
//!    chip (fallbacks over the last 100 steps). Its second line is the surface chip ([`surface_chip`]:
//!    "Shared · LSDA · h 0.15 · pseudo"; the old source line, provenance and folder key on hover)
//!    and the sample count.
//! 2. **Teach next** (open): what blocks or helps teaching — the refused autoload, the box advice,
//!    "no model yet" with Retry, Teach around blockers, the running pass's progress, the first-pass
//!    offer — then the environment keys not learned yet ([`gaps`]: the scene's first, the others
//!    under "other gaps (n)"), or "Everything this scene needs is learned"; the preset, Sample more
//!    with its cost, Cancel while a pass runs; the "Plan settings" fold; the audits line ("audits on
//!    · every N steps" / "audits off (unsafe)") and the "Audits" fold (cadence, auto, draft mode).
//! 3. **Health** (open): uncertainty against its bound, the last fit, the last audit — short.
//! 4. Folds, closed: **Learned (k classes)**, **Worst-fitting samples** (spec 2026-11-12 R2),
//!    **Details** (every readout row of the old grid, the provenance, the last pass, "What this
//!    model is (and is not)") and **Model file** (the shared collection / a model file switch,
//!    Import, Save a specialised model…, Save / Load / Forget model).
//!
//! Display-only restructuring: every control emits the `Action` it did before. Which folds are open
//! is display state ([`Sections`], `UiState::model_sections`, the profile's `model_sections`), not
//! recorded; a Ctrl K reveal unfolds the fold holding its anchor (it stays open).
use super::super::actions::{expire_arming, forget_armed, sample_button, SampleButton};
use super::super::sampling_presets as presets;
use super::super::{
    duration_label, probe_rect, search, source_colour, source_label, theme, trust, Action, PanelInput, TrustKind, UiState, AMBER, GREEN, RED, source_tip,
};
use crate::scene::model_path;
use crate::sim::{last_fit_line, source_line_with, Backend, ModelStatus, FIT_VS_HELD_HELP, IMPORT_REFERENCES_NOTE};
use egui::{vec2, Color32, RichText, Sense, Stroke, Ui};
use sim::{SamplingPlan, Source};

/// One of the tab's folds (`UiState::model_sections`). Teach next and Health start open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Section {
    Teach,
    /// "other gaps (n)" under Teach next: keys not learned whose species the scene does not hold.
    Gaps,
    /// "Plan settings" under Teach next: the sampling plan's fields.
    Plan,
    /// "Audits" under Teach next: the cadence, auto, draft mode.
    Audits,
    Health,
    Learned,
    Suspects,
    Details,
    File,
}

impl Section {
    pub const ALL: [Section; 9] =
        [Section::Teach, Section::Gaps, Section::Plan, Section::Audits, Section::Health, Section::Learned, Section::Suspects, Section::Details, Section::File];

    /// Its word in the profile's `model_sections` line (its header's probe: `model-fold-<id>`).
    pub fn id(self) -> &'static str {
        match self {
            Section::Teach => "teach",
            Section::Gaps => "gaps",
            Section::Plan => "plan",
            Section::Audits => "audits",
            Section::Health => "health",
            Section::Learned => "learned",
            Section::Suspects => "suspects",
            Section::Details => "details",
            Section::File => "file",
        }
    }

    /// A fold inside Teach next (a smaller header).
    fn sub(self) -> bool {
        matches!(self, Section::Gaps | Section::Plan | Section::Audits)
    }

    /// The search anchors its contents mark (`search::ENTRIES`): a reveal of one unfolds it.
    pub fn anchors(self) -> &'static [&'static str] {
        match self {
            Section::Teach => &["d-teach", "d-sample", "d-plan", "d-audits", "d-draft"],
            Section::Gaps => &[],
            Section::Plan => &["d-plan"],
            Section::Audits => &["d-audits", "d-draft"],
            Section::Health => &["d-model-health"],
            Section::Learned => &["d-model-learned"],
            Section::Suspects => &["d-suspects"],
            Section::Details => &["d-model-details", "d-model-about"],
            Section::File => &["d-model-file", "d-model-source"],
        }
    }
}

/// Which folds are open (`UiState::model_sections`): Teach next and Health by default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sections([bool; 9]);

impl Default for Sections {
    fn default() -> Self {
        let mut s = Sections([false; 9]);
        s.set(Section::Teach, true);
        s.set(Section::Health, true);
        s
    }
}

impl Sections {
    pub fn is_open(&self, s: Section) -> bool {
        self.0[s as usize]
    }

    pub fn set(&mut self, s: Section, open: bool) {
        self.0[s as usize] = open;
    }

    /// Every fold open (or none).
    #[cfg(test)]
    pub fn all(open: bool) -> Sections {
        Sections([open; 9])
    }

    /// The profile's value: the open folds' ids, comma separated, in panel order.
    pub fn encode(&self) -> String {
        Section::ALL.into_iter().filter(|&s| self.is_open(s)).map(Section::id).collect::<Vec<_>>().join(",")
    }

    /// The profile's value read back: the ids named are open, the rest closed; unknown words are
    /// dropped, case and spaces ignored.
    pub fn parse(v: &str) -> Sections {
        let mut s = Sections([false; 9]);
        for w in v.split(',').map(str::trim) {
            if let Some(x) = Section::ALL.into_iter().find(|x| x.id().eq_ignore_ascii_case(w)) {
                s.set(x, true);
            }
        }
        s
    }
}

/// The verdict card's tone: its border and light.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tone {
    /// The model answers here.
    Good,
    /// Not answering here (or answering unchecked): the warning colour.
    Warn,
    /// Nothing to judge yet (no model, a pass running, another solver).
    Neutral,
}

impl Tone {
    fn colour(self) -> Color32 {
        match self {
            Tone::Good => GREEN,
            Tone::Warn => AMBER,
            Tone::Neutral => theme::INK_DIM,
        }
    }
}

/// The verdict card's words ([`verdict`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Verdict {
    pub tone: Tone,
    /// One line: "Model answering · trusted here", "Falling back to DFT here" …
    pub headline: String,
    /// One short plain sentence naming the reason.
    pub reason: String,
    /// The long explanation (the "why?" hover): the old trusted / strict-here paragraphs.
    pub tip: String,
}

/// The verdict at the current geometry, from the trust view (`ui::trust`), the strict policy's
/// verdict here (`ModelStatus::readiness`), draft mode / Model only and the step's source.
pub fn verdict(st: &UiState, input: &PanelInput<'_>) -> Verdict {
    let v = |tone, headline: &str, reason: &str, tip: String| Verdict { tone, headline: headline.to_string(), reason: reason.to_string(), tip };
    if st.backend != Backend::Estimated {
        return v(Tone::Neutral, "Estimated solver off", "Pick Estimated in the toolbar's Solver group: its model, sampling pass and audits show here.", String::new());
    }
    let m = input.model();
    let d = input.snapshot.filter(|s| s.backend == Backend::Estimated).map(|s| &s.dynamics);
    if input.sampling() {
        let tip = d.and_then(|d| d.sampling.as_ref()).map_or(String::new(), |p| {
            format!("{} — {}/{}, skipped {}, failed {}. The dynamics are paused; edits are allowed.", crate::ui::activity::words::pass_stage(p), p.done, p.total, p.skipped, p.failed)
        });
        return v(Tone::Neutral, "Sampling pass running", "DFT is teaching the model; it refits when the pass ends.", tip);
    }
    let Some(m) = m.filter(|m| m.fitted) else {
        let (reason, tip) = match m.and_then(|m| m.no_model.as_deref()) {
            Some(why) => ("Samples are held but no fit formed yet: see Teach next.", why.to_string()),
            None if input.scene.nuclei.is_empty() => ("Place atoms, then Start sampling below.", String::new()),
            None => ("Every step is DFT until a sampling pass fits one: Start sampling below.", ABOUT_UNTRUSTED.to_string()),
        };
        return v(Tone::Neutral, "No model yet", reason, tip);
    };
    if let Some(why) = &m.provenance_mismatch {
        return v(
            Tone::Warn,
            "Falling back to DFT · other settings",
            "This model was taught at other settings: restore them, or forget it and sample this scene.",
            format!("{why}: every step falls back to DFT and teaches nothing — restore those settings, or forget the model and sample this scene"),
        );
    }
    let t = trust(Some(m), input.scene, st);
    let tip = [Some(t.text.clone()), t.tip.map(str::to_string), t.detail.clone().map(|d| format!("the fit's own words: {d}")), m.readiness.as_ref().map(|r| format!("strict mode here: {}", if r.answers { "answers".to_string() } else { format!("refuses — {}", r.text) }))]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join("\n\n");
    let scene_z: Vec<u8> = input.scene.nuclei.iter().map(|n| n.z).collect();
    let here: Vec<Gap> = gaps(m, &scene_z).into_iter().filter(|g| g.in_scene).collect();
    let refuses = m.readiness.as_ref().is_some_and(|r| !r.answers);
    if st.draft {
        let name = if st.model_only.is_some() { "Model only" } else { "Draft mode" };
        let unchecked = st.audit_every == 0;
        let tail = if unchecked { " and nothing checks it (audits off)" } else { "" };
        return if refuses {
            let what = if here.is_empty() { String::new() } else { format!(": {} not learned yet", gap_list(&here)) };
            Verdict { tone: Tone::Warn, headline: format!("{name} · guessing here"), reason: format!("Strict mode would refuse here{what}; the model answers anyway{tail}."), tip }
        } else {
            let reason = if unchecked { "Strict mode would answer here too, but audits are off: nothing checks the model." } else { "Strict mode would answer here too; elsewhere draft mode guesses." };
            Verdict { tone: Tone::Warn, headline: format!("{name} · model answering"), reason: reason.into(), tip }
        };
    }
    if d.map(|d| d.source) == Some(Source::Reference) {
        return v(Tone::Good, "Atom references answering", "No atom has a neighbour within the model's cutoff: the isolated-atom references answer.", source_tip(Source::Reference).to_string());
    }
    // the recent fallbacks here touch the box edge: their answers cannot be learned, whatever rule
    // sent them to DFT — the cure is the box, so it leads (2026-11-14)
    let falling_back = match t.kind {
        TrustKind::NotHere | TrustKind::ClassUntrusted | TrustKind::Stuck => true,
        TrustKind::Trusted | TrustKind::Quarantined => matches!(d.map(|d| d.source), Some(Source::Fallback | Source::Dft)),
        TrustKind::Untrusted | TrustKind::NoModel => false,
    };
    if falling_back && box_edge_fallbacks(st) {
        return Verdict { tone: Tone::Warn, headline: BOX_EDGE_HEADLINE.into(), reason: BOX_EDGE_REASON.into(), tip };
    }
    match t.kind {
        TrustKind::Untrusted => {
            let reason = if crate::sim::box_blocks_trust(m).is_some() {
                "Box-affected samples keep the model untrusted: a bigger cell is the cure (see Teach next).".to_string()
            } else {
                sentence(t.text.strip_prefix("no — ").unwrap_or(&t.text))
            };
            Verdict { tone: Tone::Warn, headline: "Falling back to DFT · model untrusted".into(), reason, tip }
        }
        TrustKind::Stuck => v(Tone::Warn, "Falling back to DFT here", "Trusted, but it has not answered at this geometry for a while.", tip),
        TrustKind::NotHere | TrustKind::ClassUntrusted if near(m).is_some() => {
            let (headline, reason) = near(m).unwrap_or_default();
            Verdict { tone: Tone::Warn, headline, reason, tip }
        }
        TrustKind::NotHere | TrustKind::ClassUntrusted => {
            let reason = if !here.is_empty() {
                format!("This scene has {} the model hasn't learned yet. Everything else here is trusted.", gap_list(&here))
            } else if m.readiness.as_ref().is_some_and(|r| r.unlearned.is_some()) {
                "This scene's own samples are not in the fit yet: Sample more here.".to_string()
            } else {
                "Strict mode refuses at this geometry: move on, or Sample more here.".to_string()
            };
            Verdict { tone: Tone::Warn, headline: "Falling back to DFT here".into(), reason, tip }
        }
        TrustKind::NoModel => v(Tone::Neutral, "No model yet", "Every step is DFT until a sampling pass fits one.", tip),
        TrustKind::Trusted | TrustKind::Quarantined => {
            let held = if m.quarantined > 0 { format!(" {} samples are held out of the fit.", m.quarantined) } else { String::new() };
            match d.map(|d| d.source) {
                Some(Source::Fallback | Source::Dft) => Verdict {
                    tone: Tone::Neutral,
                    headline: "Trusted here · this step DFT".into(),
                    reason: format!("The model was unsure at this geometry, or an audit ran: the DFT answer becomes a sample.{held}"),
                    tip,
                },
                _ => Verdict { tone: Tone::Good, headline: "Model answering · trusted here".into(), reason: format!("Everything this scene needs is learned.{held}"), tip },
            }
        }
    }
}

/// The verdict's headline when the recent fallbacks here touched the box edge (2026-11-14).
pub const BOX_EDGE_HEADLINE: &str = "Falling back to DFT here · touching the box edge";
/// Its reason.
pub const BOX_EDGE_REASON: &str = "The electron cloud reaches the grid's edge, so DFT answers here can't be learned. Move the molecule toward the centre or use a bigger grid.";
/// The verdict's headline next to samples the user forgot (2026-11-14).
pub const FORGOTTEN_FENCE_HEADLINE: &str = "Falling back to DFT here · near samples you forgot";

/// The reason next to samples the user forgot: how the fence lifts.
pub fn forgotten_fence_reason() -> String {
    format!("The model answers again once {} fresh DFT answers here agree with it — or Restore the samples.", providers::surrogate::FENCE_VOTES)
}

/// The fallbacks this many back the box-edge verdict reads (the newest first).
const BOX_EDGE_RECENT: usize = 3;

/// Whether the recent fallbacks here are "not kept: box" (`TeachOutcome::BoxNotKept`: the teacher's
/// face density above the box rule): the newest solve-log row is a fallback, and it and every
/// other fallback among the last [`BOX_EDGE_RECENT`] were not kept for the box. Read from the
/// Activity log the panel already holds.
pub fn box_edge_fallbacks(st: &UiState) -> bool {
    use sim::activity::{TeachOrigin, TeachOutcome};
    let Some(newest) = st.activity.rows.back() else { return false };
    if newest.event.origin != TeachOrigin::Fallback {
        return false;
    }
    st.activity.rows.iter().rev().filter(|r| r.event.origin == TeachOrigin::Fallback).take(BOX_EDGE_RECENT).all(|r| r.event.outcome == TeachOutcome::BoxNotKept)
}

/// A local refusal next to a sample the fit does not stand on (open-shell trust, spec 2026-11-13
/// §4.3): the strict policy refuses here by the misfit fence (`NearMisfit`) or the quarantine
/// fence (`NearQuarantine`; next to forgotten samples alone, its own words — 2026-11-14) — the
/// headline and the short reason.
fn near(m: &ModelStatus) -> Option<(String, String)> {
    let r = m.readiness.as_ref().filter(|r| !r.answers)?;
    match r.rule? {
        sim::activity::TeachRule::NearMisfit => Some((
            "Falling back to DFT here · near a misfit".into(),
            "This geometry is next to a training sample the model can't reproduce: DFT answers here.".into(),
        )),
        sim::activity::TeachRule::NearQuarantine if r.forgotten_fence => Some((FORGOTTEN_FENCE_HEADLINE.into(), forgotten_fence_reason())),
        sim::activity::TeachRule::NearQuarantine => Some((
            "Falling back to DFT here · near a quarantined sample".into(),
            "This geometry is next to one the model could not learn: DFT answers here.".into(),
        )),
        _ => None,
    }
}

/// The misfit check (`TrustKey::Misfit`): the check and its line, "3 of at most 12.3 training
/// samples it can't reproduce"; `None` before a fit.
fn misfit(m: &ModelStatus) -> Option<(&sim::activity::TrustCheck, String)> {
    let c = m.trust_checks.iter().find(|c| c.key == sim::activity::TrustKey::Misfit)?;
    Some((c, format!("{} of at most {:.1} training samples it can't reproduce", c.value as usize, c.limit)))
}

/// The misfit row's hover.
const MISFIT_TIP: &str = "the misfit set: fitted training samples the model cannot reproduce within their bound. They stay in the fit; geometries next to one are refused locally (DFT answers there, rule \"misfit\"), and the model stays trusted while they are at most 10 % of the fitted samples. The trust checklist's misfit line names the worst";

/// The untrusted model's rule, for the "no model yet" verdict's hover.
const ABOUT_UNTRUSTED: &str = "An untrusted model never answers: every step is DFT and is kept as a sample until a refit earns trust.";

/// `text` as a sentence: its first letter capitalised, a full stop at the end.
fn sentence(text: &str) -> String {
    let mut c = text.chars();
    let mut s = match c.next() {
        Some(f) => f.to_uppercase().chain(c).collect::<String>(),
        None => String::new(),
    };
    if !s.ends_with('.') {
        s.push('.');
    }
    s
}

/// One environment key not learned yet (Teach next): its short label ("pair O–S", "many-body H with
/// H, S"), its progress (`have` of `need`, for a sample or evidence floor), the short words on the
/// row's right, the full verdict line on hover, and whether the scene holds every species it names.
#[derive(Clone, Debug, PartialEq)]
pub struct Gap {
    pub label: String,
    /// Words for the verdict sentence: "O–S pairs", "H with H, S (many-body)".
    pub phrase: String,
    pub have: usize,
    pub need: usize,
    /// "12 / 60", or a short fit-error word ("fit 2.5×") when the floor is met.
    pub right: String,
    pub words: String,
    pub in_scene: bool,
}

/// "pair O–S" / "many-body H with H, S" (symbols only: Z is a key).
pub fn gap_label(class: &str, z: u8, neighbours: &[u8]) -> String {
    let sym = |z: u8| units::elements::symbol(z).to_string();
    let nb: Vec<String> = neighbours.iter().map(|&z| sym(z)).collect();
    match (class, nb.len()) {
        ("pair", 1) => format!("pair {}–{}", sym(z), nb[0]),
        (_, 0) => format!("{class} {}", sym(z)),
        _ => format!("{class} {} with {}", sym(z), nb.join(", ")),
    }
}

fn gap_phrase(class: &str, z: u8, neighbours: &[u8]) -> String {
    let sym = |z: u8| units::elements::symbol(z).to_string();
    let nb: Vec<String> = neighbours.iter().map(|&z| sym(z)).collect();
    match (class, nb.len()) {
        ("pair", 1) => format!("{}–{} pairs", sym(z), nb[0]),
        (_, 0) => format!("lone {}", sym(z)),
        _ => format!("{} with {} ({class})", sym(z), nb.join(", ")),
    }
}

/// "O–S pairs", "O–S pairs and H with H, S (many-body)", "a, b and c".
fn gap_list(g: &[Gap]) -> String {
    let p: Vec<&str> = g.iter().map(|g| g.phrase.as_str()).collect();
    match p.len() {
        0 => String::new(),
        1 => p[0].to_string(),
        n => format!("{} and {}", p[..n - 1].join(", "), p[n - 1]),
    }
}

/// The progress and the short words of a refusal: `(have, need, right)`.
fn refusal_words(refusal: Option<(sim::ClassKey, f64, f64)>, samples: usize, columns: usize) -> (usize, usize, String) {
    use sim::ClassKey as K;
    match refusal {
        Some((K::Samples, n, q)) => (n as usize, q as usize, format!("{n:.0} / {q:.0}")),
        Some((K::Window, n, w)) => (n as usize, w as usize, format!("{n:.0} / {w:.0} checks")),
        Some((K::Force, x, _)) => (samples, columns, format!("fit {x:.1}×")),
        Some((K::Energy, x, _)) => (samples, columns, format!("energy {x:.1}×")),
        Some((K::WorstAtom, x, _)) => (samples, columns, format!("DFT check {x:.1}×")),
        Some((K::Calibrated, x, _)) => (samples, columns, format!("DFT checks {x:.2}×")),
        Some((K::LooFloor, x, _)) => (samples, columns, format!("left-out {x:.2}×")),
        Some((K::MeasuredEnergy, x, _)) => (samples, columns, format!("DFT energy {x:.2}×")),
        Some((K::EnergyRms, x, _)) => (samples, columns, format!("energy {x:.1}× avg")),
        None => (samples, columns, format!("{samples} / {columns}")),
    }
}

/// The environment keys not learned yet (`ModelStatus::class_loo`, per key; a class with no keys
/// counts as one), those whose species the scene all holds first — each group in the order the
/// model lists them.
pub fn gaps(m: &ModelStatus, scene_z: &[u8]) -> Vec<Gap> {
    let mut out = Vec::new();
    for c in &m.class_loo {
        if c.species.is_empty() {
            if !c.trusted {
                let (have, need, right) = refusal_words(c.refusal, c.samples, c.samples);
                out.push(Gap { label: c.class.to_string(), phrase: format!("{} environments", c.class), have, need, right, words: crate::ui::activity::words::class_line(c), in_scene: true });
            }
            continue;
        }
        for s in c.species.iter().filter(|s| !s.trusted) {
            let (have, need, right) = refusal_words(s.refusal, s.samples, s.columns);
            let in_scene = std::iter::once(&s.z).chain(&s.neighbours).all(|z| scene_z.contains(z));
            out.push(Gap {
                label: gap_label(c.class, s.z, &s.neighbours),
                phrase: gap_phrase(c.class, s.z, &s.neighbours),
                have,
                need,
                right,
                words: crate::ui::activity::words::species_line(c.class, s),
                in_scene,
            });
        }
    }
    out.sort_by_key(|g| !g.in_scene);
    out
}

/// The learned environment keys (the Learned fold): (label, samples, the full verdict line).
fn learned(m: &ModelStatus) -> Vec<(String, usize, String)> {
    let mut out = Vec::new();
    for c in &m.class_loo {
        if c.species.is_empty() {
            if c.trusted {
                out.push((c.class.to_string(), c.samples, class_line(c)));
            }
            continue;
        }
        for s in c.species.iter().filter(|s| s.trusted) {
            out.push((gap_label(c.class, s.z, &s.neighbours), s.samples, species_line(c.class, s).trim().to_string()));
        }
    }
    out
}

/// The surface chip's words: "Shared · LSDA · h 0.15 · pseudo" — where the model comes from (the
/// shared collection, a model file, this scene or session only), the functional, h, and the ionic
/// potential (none / cores / pseudo). A shared collection's from its folder key
/// (`surrogate::SurfaceKey::slug`), else the scene's (a model taught elsewhere says so in the
/// verdict).
pub fn surface_chip(m: &ModelStatus, scene: &crate::scene::Scene) -> String {
    let ionic_word = |w: &str| match w {
        "none" => "none",
        "cores" => "cores",
        _ => "pseudo",
    };
    let from_scene = || {
        let w = match scene.ionic {
            sim::IonicPolicy::None => "none",
            sim::IonicPolicy::CoresOnly => "cores",
            _ => "pseudo",
        };
        ("LSDA".to_string(), format!("{}", scene.grid_h), w.to_string())
    };
    let (who, (f, h, ionic)) = match &m.source {
        sim::ModelSource::Collection { slug, read_only, .. } => {
            let parts: Vec<&str> = slug.split('_').collect();
            let parsed = (parts.len() >= 6).then(|| {
                let h = parts[1].strip_prefix('h').unwrap_or(parts[1]).to_string();
                let ionic = parts[parts.len() - 1].split('-').next().unwrap_or("none");
                (parts[0].to_uppercase(), h, ionic_word(ionic).to_string())
            });
            (if *read_only { "Shared (read-only)" } else { "Shared" }, parsed.unwrap_or_else(from_scene))
        }
        sim::ModelSource::File(_) => ("Model file", from_scene()),
        sim::ModelSource::None if m.collections => ("This scene", from_scene()),
        sim::ModelSource::None => ("This session", from_scene()),
    };
    format!("{who} · {f} · h {h} · {ionic}")
}

pub fn show(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    expire_arming(st);
    // the visible part of the panel (the tests' width checks)
    probe_rect("model-panel", ui.clip_rect());
    // a Ctrl K reveal unfolds the folds holding its anchor until it has scrolled into view; they stay open
    if let Some(a) = st.search.unrevealed() {
        for s in Section::ALL.into_iter().filter(|s| s.anchors().contains(&a)) {
            st.model_sections.set(s, true);
        }
    }
    resume_row(ui, st, input, acts);
    super::selection::segment_row(ui, st, input, "d-model-segment", acts);
    let m: Option<&ModelStatus> = input.model();
    let snap = input.snapshot.filter(|s| s.backend == Backend::Estimated);
    let d = snap.map(|s| &s.dynamics);
    super::group(ui, st, "d-model", |ui, st| verdict_card(ui, st, input, m, d, acts));
    if st.backend != Backend::Estimated {
        return;
    }
    let sampling = input.sampling();
    ui.add_space(2.0);
    // 2. Teach next
    let gs = m.map(|m| gaps(m, &input.scene.nuclei.iter().map(|n| n.z).collect::<Vec<_>>())).unwrap_or_default();
    let fitted = m.is_some_and(|m| m.fitted);
    let teach_summary = if !fitted {
        None
    } else if gs.iter().any(|g| g.in_scene) {
        let n = gs.iter().filter(|g| g.in_scene).count();
        Some((format!("{n} to learn here"), AMBER))
    } else {
        Some(("all learned here".to_string(), GREEN))
    };
    fold(ui, st, Section::Teach, "Teach next", teach_summary, Some("model-teach"), |ui, st| teach(ui, st, input, m, d, &gs, sampling, acts));
    // 3. Health
    let health_summary = d.and_then(|d| d.uncertainty).zip(m).map(|(u, m)| if u <= m.force_bound { ("within bound".to_string(), GREEN) } else { ("above bound".to_string(), AMBER) });
    fold(ui, st, Section::Health, "Health", health_summary, Some("model-health"), |ui, st| {
        super::group(ui, st, "d-model-health", |ui, _| health(ui, m, d));
    });
    // 4. the folds
    let learned_rows = m.map(learned).unwrap_or_default();
    let learned_summary = (!learned_rows.is_empty()).then(|| (format!("✓ {} learned", learned_rows.len()), GREEN));
    let title = format!("Learned ({} {})", learned_rows.len(), if learned_rows.len() == 1 { "class" } else { "classes" });
    fold(ui, st, Section::Learned, &title, learned_summary, None, |ui, st| {
        super::group(ui, st, "d-model-learned", |ui, _| learned_fold(ui, &learned_rows));
    });
    let worst = m.and_then(|m| m.suspects.first()).map(|s| {
        let r = s.loo_ratio.map_or("not in the fit".to_string(), |r| format!("{}×", ratio_words(r)));
        (format!("#{} {} {r}", s.id.index, s.formula), if s.loo_ratio.is_some_and(|r| !(r <= 1.0)) { AMBER } else { theme::INK_DIM })
    });
    // spec 2026-11-12 R2: the fit's worst-fitting samples, each with Forget; the forgotten ones with Restore
    fold(ui, st, Section::Suspects, "Worst-fitting samples", worst, None, |ui, st| {
        super::group(ui, st, "d-suspects", |ui, st| suspects(ui, st, m, sampling, acts));
    });
    let details_summary = m.map(|m| (format!("epoch {} · {} samples", m.epoch, m.samples), theme::INK_DIM));
    fold(ui, st, Section::Details, "Details", details_summary, None, |ui, st| details(ui, st, input, m, d, sampling));
    let file_summary = m.map(|m| {
        let w = match &m.source {
            sim::ModelSource::Collection { read_only: true, .. } => "shared · read-only".to_string(),
            sim::ModelSource::Collection { .. } => "shared".to_string(),
            sim::ModelSource::File(p) => source_line_with(m, st.model_file.as_deref()).strip_prefix("file: ").map_or_else(|| p.display().to_string(), str::to_string),
            sim::ModelSource::None => "not saved".to_string(),
        };
        (w, theme::INK_DIM)
    });
    fold(ui, st, Section::File, "Model file", file_summary, None, |ui, st| {
        let scene_z: Vec<u8> = input.scene.nuclei.iter().map(|n| n.z).collect();
        super::group(ui, st, "d-model-source", |ui, st| source_row(ui, st, m, sampling, &scene_z, acts));
        super::group(ui, st, "d-model-file", |ui, st| {
            model_file_buttons(ui, st, input, acts);
            ui.label(RichText::new(model_path(&st.scene_path).display().to_string()).monospace().size(11.0).color(theme::INK_DIM));
        });
    });
}

/// A fold's header row (the View panel's: a caret, its title; a click folds or unfolds it) with a
/// one-line summary on its right, cut with "…" when the panel is narrow. Returns whether it is open.
fn header(ui: &mut Ui, st: &mut UiState, s: Section, title: &str, summary: Option<(String, Color32)>) -> bool {
    let (r, _) = theme::fold_header(ui, title, st.model_sections.is_open(s), summary, None, s.sub());
    probe_rect(&format!("model-fold-{}", s.id()), r.rect);
    if r.clicked() {
        let open = st.model_sections.is_open(s);
        st.model_sections.set(s, !open);
    }
    st.model_sections.is_open(s)
}

/// A fold: its header and, while open, its body (indented); `probe` records the whole fold's rect.
fn fold(ui: &mut Ui, st: &mut UiState, s: Section, title: &str, summary: Option<(String, Color32)>, probe: Option<&str>, body: impl FnOnce(&mut Ui, &mut UiState)) {
    let r = ui.scope(|ui| {
        if header(ui, st, s, title, summary) {
            ui.indent(("model-fold", s.id()), |ui| body(ui, st));
            ui.add_space(4.0);
        }
    });
    if let Some(p) = probe {
        probe_rect(p, r.response.rect);
    }
}

/// A small rounded chip (the surface chip, the "N % DFT" chip).
fn chip(ui: &mut Ui, text: &str, colour: Color32) -> egui::Response {
    egui::Frame::new()
        .fill(colour.gamma_multiply(0.10))
        .stroke(Stroke::new(1.0, colour.gamma_multiply(0.6)))
        .corner_radius(egui::CornerRadius::same(255))
        .inner_margin(egui::Margin::symmetric(7, 1))
        .show(ui, |ui| ui.add(egui::Label::new(RichText::new(text).size(11.0).color(colour)).wrap()))
        .response
}

/// The text's width at a chip's font, with the chip's margins.
fn chip_width(ui: &Ui, text: &str) -> f32 {
    ui.painter().layout_no_wrap(text.to_string(), egui::FontId::proportional(11.0), theme::INK).size().x + 16.0
}

/// 1. The verdict card ([`verdict`]): the headline with its light and the "N % DFT" chip, the reason
/// with "why?" (the long explanation on hover), the residual check's Forget, the surface chip and the
/// sample count.
fn verdict_card(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, m: Option<&ModelStatus>, d: Option<&sim::DynSnapshot>, acts: &mut Vec<Action>) {
    let v = verdict(st, input);
    let c = v.tone.colour();
    let (fill, stroke) = match v.tone {
        Tone::Neutral => (theme::PANEL_2, theme::LINE),
        _ => (c.gamma_multiply(0.06), c.gamma_multiply(0.55)),
    };
    let r = egui::Frame::new().fill(fill).stroke(Stroke::new(1.0, stroke)).corner_radius(8).inner_margin(egui::Margin::symmetric(10, 8)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        let pct = d.filter(|_| st.backend == Backend::Estimated && m.is_some()).map(|d| format!("{:.0} % DFT", 100.0 * d.fallback_fraction_100));
        ui.horizontal_top(|ui| {
            theme::light(ui, c);
            let cw = pct.as_deref().map_or(0.0, |p| chip_width(ui, p) + ui.spacing().item_spacing.x);
            let w = (ui.available_width() - cw).max(80.0);
            ui.allocate_ui_with_layout(vec2(w, 0.0), egui::Layout::top_down(egui::Align::Min), |ui| {
                ui.set_width(w);
                super::super::probed("model-headline", ui.add(egui::Label::new(RichText::new(&v.headline).strong().size(13.5).color(theme::INK)).wrap()));
            });
            if let (Some(p), Some(d)) = (&pct, d) {
                let pc = if d.fallback_fraction_100 > 0.0 { AMBER } else { theme::INK_DIM };
                super::super::probed("model-dft-chip", chip(ui, p, pc)).on_hover_text("fallbacks over the last 100 steps: the share of steps DFT answered");
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.add(egui::Label::new(RichText::new(&v.reason).size(12.0).color(theme::INK_DIM)).wrap());
            if !v.tip.is_empty() {
                super::super::probed("model-why", ui.label(RichText::new("why?").size(11.5).underline().color(theme::ACCENT_INK))).on_hover_text(&v.tip);
            }
        });
        // the failed residual check's worst sample, with the Worst-fitting rows' Forget
        if let Some(id) = m.filter(|m| m.fitted).and_then(|m| trust(Some(m), input.scene, st).forget) {
            if super::super::probed(&format!("model-trust-forget-{}", id.index), ui.small_button(format!("Forget #{}", id.index)))
                .on_hover_text(crate::ui::activity::trust::FORGET_WORST_TIP)
                .clicked()
            {
                acts.push(Action::ForgetSample(id));
            }
        }
        if let Some(m) = m.filter(|_| st.backend == Backend::Estimated) {
            let words = surface_chip(m, input.scene);
            let read_only = matches!(m.source, sim::ModelSource::Collection { read_only: true, .. });
            let samples = format!("{} samples", m.samples);
            let mut tip = format!("{}\n\n{SOURCE_LINE_TIP}", source_line_with(m, st.model_file.as_deref()));
            if let Some(p) = &m.provenance {
                tip += &format!("\n\ntaught at {p}");
            }
            let fits = chip_width(ui, &words) + 70.0 < ui.available_width();
            let draw = |ui: &mut Ui| {
                super::super::probed("model-source", chip(ui, &words, if read_only { AMBER } else { theme::INK_DIM })).on_hover_text(&tip);
            };
            if fits {
                ui.horizontal(|ui| {
                    draw(ui);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| ui.label(RichText::new(&samples).size(11.5).color(theme::INK_DIM)));
                });
            } else {
                draw(ui);
                ui.label(RichText::new(&samples).size(11.5).color(theme::INK_DIM));
            }
        }
    });
    probe_rect("model-verdict", r.response.rect);
}

/// One gap row: its label (wrapping), a small progress bar of `have` / `need` and the short words.
fn gap_row(ui: &mut Ui, g: &Gap) {
    ui.horizontal_top(|ui| {
        let rw = ui.painter().layout_no_wrap(g.right.clone(), egui::FontId::monospace(11.0), theme::INK).size().x;
        let bar = 52.0;
        let w = (ui.available_width() - bar - rw - 2.0 * ui.spacing().item_spacing.x).max(60.0);
        let r = ui.allocate_ui_with_layout(vec2(w, 0.0), egui::Layout::top_down(egui::Align::Min), |ui| {
            ui.set_width(w);
            ui.add(egui::Label::new(RichText::new(&g.label).size(12.0).color(theme::INK)).wrap())
        });
        r.inner.on_hover_text(&g.words);
        let (rect, _) = ui.allocate_exact_size(vec2(bar, 14.0), Sense::hover());
        let track = egui::Rect::from_center_size(rect.center(), vec2(bar, 5.0));
        ui.painter().rect_filled(track, 2.5, theme::PANEL_3);
        let f = if g.need > 0 { (g.have as f32 / g.need as f32).clamp(0.0, 1.0) } else { 1.0 };
        let mut fill = track;
        fill.set_width(track.width() * f);
        ui.painter().rect_filled(fill, 2.5, AMBER);
        ui.label(RichText::new(&g.right).monospace().size(11.0).color(theme::INK_DIM)).on_hover_text(&g.words);
    });
}

/// 2. Teach next: what blocks teaching and its fixes, the keys not learned yet, the sampling
/// controls (preset, Sample more, the cost, Cancel while a pass runs), the plan's fold, the audits
/// line and its fold.
#[allow(clippy::too_many_arguments)]
fn teach(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, m: Option<&ModelStatus>, d: Option<&sim::DynSnapshot>, gs: &[Gap], sampling: bool, acts: &mut Vec<Action>) {
    super::group(ui, st, "d-teach", |ui, st| {
        if let Some(m) = m {
            if let Some(r) = &m.autoload_refused {
                warnbox(ui, &format!("this session's model was not restored: {r}"));
                if ui.button(RichText::new("Forget it and sample this scene").color(AMBER)).clicked() {
                    acts.push(Action::SampleAfterRefusal);
                }
            }
            if crate::sim::box_blocks_trust(m).is_some() {
                warnbox(ui, &super::super::box_advice(input.scene, st));
            }
            no_model_row(ui, m, sampling, acts);
            teach_row(ui, m, sampling, acts);
        }
        if let Some(p) = d.and_then(|d| d.sampling.as_ref()).filter(|_| sampling) {
            progress(ui, p, m);
        }
        if let Some(o) = m.and_then(|m| m.pass_offer.as_ref()).filter(|_| !sampling) {
            offer(ui, o, acts);
            lone_note(ui, input, "lone-note-offer");
        }
        if m.is_some_and(|m| m.fitted) {
            let (here, other): (Vec<&Gap>, Vec<&Gap>) = gs.iter().partition(|g| g.in_scene);
            if here.is_empty() {
                super::super::probed("model-all-learned", ui.label(RichText::new("✓ Everything this scene needs is learned").size(12.0).color(GREEN)));
            }
            for g in &here {
                gap_row(ui, g);
            }
            if !other.is_empty() {
                let title = format!("other gaps ({})", other.len());
                fold(ui, st, Section::Gaps, &title, None, None, |ui, _| {
                    for g in &other {
                        gap_row(ui, g);
                    }
                });
            }
        }
    });
    ui.add_space(2.0);
    super::group(ui, st, "d-plan", |ui, st| preset_row(ui, &mut st.plan));
    super::group(ui, st, "d-sample", |ui, st| sample_row(ui, st, input, acts));
    super::group(ui, st, "d-plan", |ui, st| {
        fold(ui, st, Section::Plan, "Plan settings", None, None, |ui, st| {
            if let Some(i) = presets::matching(&st.plan) {
                theme::note(ui, presets::PRESETS[i].description);
            }
            let tu = st.temp_units;
            plan_fields(ui, &mut st.plan, tu);
        });
    });
    super::group(ui, st, "d-audits", |ui, st| {
        audits_line(ui, st, m, acts);
        fold(ui, st, Section::Audits, "Audits", None, None, |ui, st| {
            audit_controls(ui, st, acts);
            // Q1 of Appendix B (M5b): draft mode, also in Physics ▸ Estimated solver
            super::group(ui, st, "d-draft", |ui, st| {
                theme::section(ui, "Draft mode");
                super::super::menus::draft_control(ui, st, acts);
            });
        });
    });
}

/// The audits in one line: "audits on · every N steps" (the switch, with why they matter on hover),
/// or the unsafe "audits off (unsafe)" whose switch turns them back on (the last cadence restored).
fn audits_line(ui: &mut Ui, st: &mut UiState, m: Option<&ModelStatus>, acts: &mut Vec<Action>) {
    ui.horizontal_wrapped(|ui| {
        let mut on = st.audit_every > 0;
        let text = if on { RichText::new("audits on") } else { RichText::new("audits off (unsafe)").color(AMBER) };
        // an on/off switch (user request 2026-10-18), with why they matter
        let r = super::super::probed("audits-on", ui.checkbox(&mut on, text)).on_hover_text(if st.audit_every > 0 { AUDIT_WHY } else { AUDIT_OFF_NOTE });
        if r.changed() {
            if on {
                acts.push(Action::SetAudit(st.audit_every_on.max(1)));
            } else {
                st.audit_every_on = st.audit_every.max(1);
                acts.push(Action::SetAudit(0));
            }
        }
        if st.audit_every > 0 {
            ui.label(RichText::new(format!("· every {} steps{}", st.audit_every, if st.audit_auto { " (auto)" } else { "" })).size(12.0).color(theme::INK_DIM));
        }
        // what the worker runs (after a panic recovery it re-applies this value)
        if let Some(m) = m.filter(|m| m.audit_every != st.audit_every) {
            ui.colored_label(AMBER, format!("(solver: {})", m.audit_every));
        }
    });
}

/// The Audits fold: the cadence ("DFT audit every n answered steps") and auto; while off, the
/// unsafe note.
fn audit_controls(ui: &mut Ui, st: &mut UiState, acts: &mut Vec<Action>) {
    ui.horizontal_wrapped(|ui| {
        ui.label("DFT audit every");
        let mut n = st.audit_every;
        if ui
            .add(egui::DragValue::new(&mut n).range(0..=1_000_000).speed(10.0))
            .on_hover_text("every n-th step the model answers, the DFT teacher checks it at that geometry and the gap is recorded (a failed audit untrusts the model until a refit). 2000 by default. With a GPU teacher the audit runs on a second teacher in the background while the model keeps stepping (ruling S14), so the cadence costs no speed: the check simply arrives a few hundred steps later. With the CPU teacher it runs in the step and sets the speed. 0 = never (unsafe: nothing checks the model while it answers).")
            .changed()
        {
            acts.push(Action::SetAudit(n));
        }
        ui.label("answered steps");
        super::super::probed("audit-auto", ui.checkbox(&mut st.audit_auto, "auto")).on_hover_text(format!(
            "set the cadence from the play speed so an audit comes about every {:.0} s of wall time at any speed (e.g. ≈ 15 fs/s at dt 0.25 fs → every ≈ 600 steps, raised to the floor; ≈ 4000 fs/s → every ≈ 160 000). Faster play checks a smaller share of steps but keeps the same steady load on the GPU; the cadence stays between {} and {} steps",
            super::super::AUTO_AUDIT_WALL_S, super::super::AUTO_AUDIT_MIN, super::super::AUTO_AUDIT_MAX
        ));
    });
    if st.audit_every == 0 {
        ui.colored_label(AMBER, "audits off: nothing checks the model against DFT while it answers (unsafe)");
        ui.label(RichText::new(AUDIT_OFF_NOTE).size(11.0).color(theme::INK_DIM));
    }
}

/// A label / value row of Health and Details: the label in a fixed dim column, the value wrapping
/// in the rest of the width.
fn row<R>(ui: &mut Ui, label: &str, value: impl FnOnce(&mut Ui) -> R) -> R {
    ui.horizontal_top(|ui| {
        let w = 92.0f32.min(ui.available_width() * 0.36);
        ui.allocate_ui_with_layout(vec2(w, 0.0), egui::Layout::top_down(egui::Align::Min), |ui| {
            ui.set_width(w);
            ui.add(egui::Label::new(RichText::new(label).size(12.0).color(theme::INK_DIM)).wrap());
        });
        ui.vertical(|ui| {
            ui.set_width(ui.available_width());
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
            value(ui)
        })
        .inner
    })
    .inner
}

/// The uncertainty row's hover.
const UNCERTAINTY_TIP: &str = "the model's epistemic force std per component (max over atoms; the label noise σ_f is left out — a fallback on it would teach nothing, and a fit whose σ_f exceeds 2e-3 Ha/Bohr is untrusted instead) against the tightest per-atom bound min(max(2e-3, 10 % |F|), 1e-2) Ha/Bohr. A MODEL uncertainty: the surrogate inherits every DFT approximation (smearing, LSDA) on top of it.";

/// The last audit's colour, its full line (Details) and its short form (Health).
fn audit_words(m: &ModelStatus) -> (Color32, String, String) {
    // the step it was judged at (the driver's stamp on its event; activity Task 5): for a
    // background audit that is after its dispatch, by the lag in answered steps
    let at = m.last_audit_step.map_or(String::new(), |n| format!(" at step {n}"));
    match (&m.last_audit, m.audit_in_flight) {
        (None, true) => (Color32::GRAY, "in flight on the background teacher".into(), "in flight".into()),
        (Some(a), in_flight) => {
            let (c, verdict) = match a.verdict {
                sim::AuditVerdict::Passed => (GREEN, "passed"),
                sim::AuditVerdict::Failed => (AMBER, "failed"),
                sim::AuditVerdict::BoxAffected => (AMBER, "not judged (box-affected)"),
            };
            let lag = if a.asynchronous { format!("; audit lag: {} answered steps", a.unaudited_steps) } else { String::new() };
            let next = if in_flight { "; next in flight" } else { "" };
            (c, format!("{verdict}: model off by {:+.2e} Ha{at} ({:.0} ms DFT{lag}{next})", a.gap, a.teacher_ms), format!("{verdict} · {:+.1e} Ha{at}", a.gap))
        }
        (None, false) => (Color32::GRAY, "— (none yet)".into(), "— (none yet)".into()),
    }
}

fn audit_tip(m: &ModelStatus) -> &'static str {
    if m.audits_background {
        "E_DFT − E_model at the last audited geometry. The audit runs on a second GPU teacher in the background while the model keeps answering (ruling S14); a failed one untrusts the model when it arrives. Audit lag = answered steps between the audit's dispatch and its verdict: those steps were answered unaudited. At full speed the model outruns the background teacher, so audits run back to back — one per teacher solve, whatever the 2000-step cadence says — and the lag is tens of thousands of steps. A box-affected teacher answer (density on the outermost grid plane > 2e-5 × (h / 0.25 Bohr)² e/Bohr³) carries the box's confinement and is not judged."
    } else {
        "E_DFT − E_model at the last audited geometry. With the CPU teacher the audit runs in the step (a second CPU solver would compete for the same cores). A box-affected teacher answer (density on the outermost grid plane > 2e-5 × (h / 0.25 Bohr)² e/Bohr³) carries the box's confinement and is not judged."
    }
}

/// 3. Health: the uncertainty against its bound (a bar with the bound marked, and the numbers), the
/// last fit (rmse, the refit's time) and the last audit, short; the full rows are in Details.
fn health(ui: &mut Ui, m: Option<&ModelStatus>, d: Option<&sim::DynSnapshot>) {
    let Some(m) = m else {
        theme::note(ui, "no model status yet");
        return;
    };
    row(ui, "uncertainty", |ui| match d.and_then(|d| d.uncertainty) {
        Some(u) => {
            let c = if u <= m.force_bound { GREEN } else { AMBER };
            uncertainty_bar(ui, u, m.force_bound, c);
            ui.colored_label(c, RichText::new(format!("{u:.1e} / {:.1e} Ha/Bohr", m.force_bound)).monospace().size(11.5)).on_hover_text(UNCERTAINTY_TIP);
        }
        None => {
            ui.colored_label(Color32::GRAY, "— (model not consulted)").on_hover_text(UNCERTAINTY_TIP);
        }
    });
    row(ui, "last fit", |ui| {
        let refit = m.last_refit.as_ref().map_or(String::new(), |r| match &r.result {
            Ok(_) => format!(" · refit {:.0} ms", r.ms),
            Err(_) => " · last refit failed".to_string(),
        });
        match m.fit {
            Some(fit @ (_, _, _, f)) => ui.label(RichText::new(format!("rmse {f:.1e} Ha/Bohr{refit}")).monospace().size(11.5)).on_hover_text(last_fit_line(fit, m.body4_degree)),
            None => ui.colored_label(Color32::GRAY, "—"),
        };
    });
    row(ui, "last audit", |ui| {
        let (c, full, short) = audit_words(m);
        ui.colored_label(c, RichText::new(short).size(11.5)).on_hover_text(format!("{full}\n\n{}", audit_tip(m)));
    });
    // samples the fit cannot reproduce: here only when there are some (else in Details)
    if let Some((c, line)) = misfit(m).filter(|(c, _)| c.value > 0.0) {
        row(ui, "misfit", |ui| ui.colored_label(if c.met { theme::INK_DIM } else { AMBER }, RichText::new(line).size(11.5)).on_hover_text(MISFIT_TIP));
    }
}

/// The Learned fold: one row per learned environment key (its samples on the right, its verdict
/// line and the class rule on hover).
fn learned_fold(ui: &mut Ui, rows: &[(String, usize, String)]) {
    if rows.is_empty() {
        theme::note(ui, "nothing learned yet: the classes are judged after the first fit");
        return;
    }
    for (label, n, words) in rows {
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("✓ {label}")).size(12.0).color(GREEN)).on_hover_text(format!("{words}\n\n{CLASS_LOO_HELP}"));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| ui.label(RichText::new(format!("{n} samples")).size(11.5).color(theme::INK_DIM)));
        });
    }
}

/// The Details fold: every readout row of the old grid ([`readouts`]), then where the model comes
/// from, what it was taught at, the last pass, and "What this model is (and is not)".
fn details(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, m: Option<&ModelStatus>, d: Option<&sim::DynSnapshot>, sampling: bool) {
    super::group(ui, st, "d-model-details", |ui, st| {
        if let (Some(d), Some(m)) = (d, m) {
            readouts(ui, d, m, input.scene, st);
        }
        if let Some(m) = m {
            row(ui, "model", |ui| {
                ui.label(RichText::new(source_line_with(m, st.model_file.as_deref())).size(11.5)).on_hover_text(SOURCE_LINE_TIP);
            });
            // the model's info line (M6 Task 5's parked item): what it was taught at, and the mass
            // set recorded with it — recorded, never compared (ruling P2)
            if let Some(p) = &m.provenance {
                theme::note(ui, format!("taught at {p}"))
                    .on_hover_text("the settings a model is valid at (grid, h, σ, charge, multiplicity, teacher and its tolerance — a scene at others falls back to DFT), and the isotopes its samples were taken with: recorded, not compared, because the Born-Oppenheimer surface does not depend on the nuclear masses");
            }
            if let Some(why) = &m.provenance_mismatch {
                warnbox(ui, &format!("{why}: every step falls back to DFT and teaches nothing — restore those settings, or forget the model and sample this scene"));
            }
        }
        // the last sampling pass
        if let Some(p) = d.and_then(|d| d.sampling.as_ref()).filter(|_| !sampling) {
            theme::note(ui, format!("last pass {}: {}/{} jobs, skipped {}, failed {}", if p.cancelled { "cancelled" } else { "finished" }, p.done, p.total, p.skipped, p.failed));
        }
    });
    let r = egui::CollapsingHeader::new("What this model is (and is not)")
        .id_salt("model-about")
        .default_open(false)
        .open(search::opens(st, "d-model-about"))
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                theme::note(ui, "estimated = a model fitted to this scene's DFT; valid only where it has been taught — it falls back to DFT when unsure");
                theme::info(ui, "The estimated solver is a model of this DFT, not of nature: it inherits every approximation of its teacher (smearing, LSDA, the box) and adds its own fit error. Help › Physics, honestly has the long form.");
            });
            theme::note(ui, "A MODEL uncertainty: the surrogate inherits every DFT approximation (smearing, LSDA) on top of it. An untrusted model never answers: every step is DFT and is kept as a sample until a refit earns trust.");
            theme::note(ui, "Box-affected teacher answers (density on the outermost grid plane above 2e-5 × (h / 0.25 Bohr)² e/Bohr³) are used for that step but never trained on.");
        });
    search::mark(ui, st, "d-model-about", r.header_response.rect);
}

/// Where the model learns (M8a Task 4b; the Model file fold): the switch between the
/// shared collection and a model file (offered when a collections folder is set), and "Import a
/// model file…" with the last import's outcome.
fn source_row(ui: &mut Ui, st: &mut UiState, m: Option<&ModelStatus>, sampling: bool, scene_z: &[u8], acts: &mut Vec<Action>) {
    let Some(m) = m else { return };
    // the switch shows the user's choice (a charged scene learns per-scene under it); Import needs
    // the collection attached
    let shared = m.use_collection;
    let attached = matches!(m.source, sim::ModelSource::Collection { .. });
    let read_only = matches!(m.source, sim::ModelSource::Collection { read_only: true, .. });
    // the source line itself is the verdict card's surface chip (its hover) and a Details row
    if !m.collections {
        return;
    }
    ui.horizontal_wrapped(|ui| {
        if ui.add_enabled(!sampling, egui::RadioButton::new(shared, "shared collection")).on_hover_text("learn into the one model of this solver surface (h, σ, ionic potential, teacher): every scene and box at these settings adds to it, and it saves itself as it learns (models/collections)").clicked() && !shared {
            acts.push(Action::UseCollection);
        }
        if ui.add_enabled(!sampling, egui::RadioButton::new(!shared, "a model file…")).on_hover_text("switch to a model file as a specialised model for this scene (per-scene; Save model writes it)").clicked() {
            let last = st.model_file.clone().unwrap_or_else(|| model_path(&st.scene_path));
            if let Some(p) = model_dialog(&last).pick_file() {
                st.model_file = Some(p);
                acts.push(Action::LoadModel);
            }
        }
    });
    if attached {
        specialise_row(ui, st, m, scene_z, acts);
    }
    let can_import = attached && !read_only && !sampling;
    if super::super::probed("import-model", ui.add_enabled(can_import, egui::Button::new("Import a model file…")))
        .on_disabled_hover_text("needs the shared collection attached and writable, and no pass running")
        .on_hover_text(format!("add a .model file's samples to the shared collection — each one checked (the same solver surface, box status known, not a duplicate); what is refused is listed with the reason. {IMPORT_REFERENCES_NOTE}"))
        .clicked()
    {
        let last = st.model_file.clone().unwrap_or_else(|| model_path(&st.scene_path));
        if let Some(p) = model_dialog(&last).pick_file() {
            acts.push(Action::ImportModel(p));
        }
    }
    match &m.last_import {
        Some(Ok(r)) => {
            ui.label(RichText::new(import_line(r)).size(11.5));
            for (i, why) in r.refused.iter().take(5) {
                ui.weak(format!("  {}: {why}", i.map_or("the file".to_string(), |i| format!("sample #{i}"))));
            }
            if r.refused.len() > 5 {
                ui.weak(format!("  … and {} more refused", r.refused.len() - 5));
            }
            for n in r.notes.iter().take(3) {
                ui.weak(n);
            }
            ui.label(RichText::new(IMPORT_REFERENCES_NOTE).size(11.0).color(theme::INK_DIM));
        }
        Some(Err(e)) => warnbox(ui, &format!("import refused: {e}")),
        None => {}
    }
}

/// "Save a specialised model…" (M8a Task 7; spec §4.2): a species picker — the scene's species
/// preselected, plus every species the shared model knows — and the file dialog; the save is
/// `Action::SaveSpecialised`. The file loads through "a model file…" (any neutral scene of this
/// solver surface, whatever its box).
fn specialise_row(ui: &mut Ui, st: &mut UiState, m: &ModelStatus, scene_z: &[u8], acts: &mut Vec<Action>) {
    let mut scene: Vec<u8> = scene_z.to_vec();
    scene.sort_unstable();
    scene.dedup();
    if st.specialise.1 != scene {
        st.specialise = (scene.clone(), scene.clone());
    }
    let mut offered: Vec<u8> = scene.iter().copied().chain(m.model_species.iter().copied()).collect();
    offered.sort_unstable();
    offered.dedup();
    egui::CollapsingHeader::new(RichText::new("Save a specialised model…").size(12.5)).id_salt("specialise").show(ui, |ui| {
        ui.weak("a model fitted only on the shared samples made of these species: smaller and faster; it loads as a model file in any scene of this solver surface");
        ui.horizontal_wrapped(|ui| {
            for z in &offered {
                let mut on = st.specialise.0.contains(z);
                if ui.checkbox(&mut on, units::elements::symbol(*z)).changed() {
                    st.specialise.0.retain(|x| x != z);
                    if on {
                        st.specialise.0.push(*z);
                        st.specialise.0.sort_unstable();
                    }
                }
            }
        });
        let can = !st.specialise.0.is_empty();
        if super::super::probed("save-specialised", ui.add_enabled(can, egui::Button::new("Save…"))).on_disabled_hover_text("tick at least one species").clicked() {
            let last = st.model_file.clone().unwrap_or_else(|| model_path(&st.scene_path));
            if let Some(p) = model_dialog(&last).save_file() {
                acts.push(Action::SaveSpecialised(p));
            }
        }
    });
}

/// A leave-one-out ratio as the "Worst-fitting samples" rows round it (as the recovery note does):
/// a whole number from 10×, one decimal below, "∞" when undefined (the sample alone spans a
/// direction of the fit).
pub fn ratio_words(r: f64) -> String {
    if !r.is_finite() {
        "∞".to_string()
    } else if r >= 10.0 {
        format!("{r:.0}")
    } else {
        format!("{r:.1}")
    }
}

/// One "Worst-fitting samples" row (spec R2): `#1804 · HO · fallback · 38× its bound` — the store
/// index, the formula from Z, the origin from the sample's tag, and how far the fit without it
/// predicts it over its own bound; "not in the fit" for a forgotten sample the fit that stands
/// does not hold.
pub fn suspect_line(s: &sim::SampleSuspect) -> String {
    let ratio = match s.loo_ratio {
        Some(r) => format!("{}× its bound", ratio_words(r)),
        None => "not in the fit".to_string(),
    };
    format!("#{} · {} · {} · {ratio}", s.id.index, s.formula, s.origin)
}

/// Below the list (spec R2).
pub const SUSPECTS_CAPTION: &str = "Ranked by how badly the fit without each sample predicts it.";
/// The section without a model.
pub const SUSPECTS_NO_MODEL: &str = "no model yet: the samples are ranked once a fit stands";
/// The section with a model whose ranking is not computed yet.
pub const SUSPECTS_NO_RANKING: &str = "no ranking yet: it is computed with the next refit";
/// The hover of a row's Forget.
const FORGET_SAMPLE_TIP: &str = "hold this sample out of the fit until you restore it: geometries near it fall back to DFT, and the trust recovery never releases it (it counts with the quarantined samples against the 10 % the recovery may hold out). Written at once — to the shared collection when it holds the sample — then the model refits without it";

/// "Purge forgotten (N)…"'s inline confirmation (2026-11-14).
pub fn purge_confirm_text(n: usize) -> String {
    format!(
        "Delete {n} forgotten sample{} from the shared collection for good? A backup of its samples file is kept as samples.bin.bak (replacing an older backup). This can't be undone from the app.",
        if n == 1 { "" } else { "s" }
    )
}
/// The hover of "Purge forgotten (N)…".
const PURGE_TIP: &str = "delete the forgotten samples for good — from the shared collection's samples file too (its old file is kept as samples.bin.bak), or, with a model file, from the model (Save model then writes it without them). Only the samples you forgot: held, superseded, box-affected and draft samples stay. The fit is unchanged — they are already out of it — and later samples are renumbered";

/// The "Worst-fitting samples" section's body (spec R2): the fit's worst samples by their
/// leave-one-out ratio (`ModelStatus::suspects`), each with **Forget** (`Action::ForgetSample`),
/// the caption, and the forgotten samples, each with **Restore** (`Action::RestoreSample`); with no
/// model, or no ranking yet, it says why. The forgotten list's header carries **Purge forgotten
/// (N)…** (2026-11-14; hidden with none forgotten, disabled with the reason during a pass or a
/// refit or when the provider refuses — `ModelStatus::purge_refusal`): it opens an inline
/// confirmation whose **Purge** emits `Action::PurgeForgotten` and whose **Cancel** emits nothing.
pub fn suspects(ui: &mut Ui, st: &mut UiState, m: Option<&ModelStatus>, sampling: bool, acts: &mut Vec<Action>) {
    match m {
        None => {
            ui.weak(SUSPECTS_NO_MODEL);
        }
        Some(m) if !m.fitted => {
            ui.weak(SUSPECTS_NO_MODEL);
        }
        Some(m) if m.suspects.is_empty() => {
            ui.weak(SUSPECTS_NO_RANKING);
        }
        Some(m) => {
            for s in &m.suspects {
                // wrapped: a narrow panel puts Forget under the row instead of past the edge
                ui.horizontal_wrapped(|ui| {
                    let worst = s.loo_ratio.is_some_and(|r| !(r <= 1.0));
                    ui.label(RichText::new(suspect_line(s)).monospace().size(11.5).color(if worst { AMBER } else { theme::INK }));
                    if super::super::probed(&format!("forget-sample-{}", s.id.index), ui.small_button("Forget")).on_hover_text(FORGET_SAMPLE_TIP).clicked() {
                        acts.push(Action::ForgetSample(s.id));
                    }
                });
            }
            ui.label(RichText::new(SUSPECTS_CAPTION).size(11.0).color(theme::INK_DIM));
        }
    }
    let Some(m) = m.filter(|m| !m.forgotten_samples.is_empty()) else {
        st.purge_confirm = false;
        return;
    };
    let n = m.forgotten_samples.len();
    ui.add_space(4.0);
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(format!("forgotten ({n}): held out of the fit until restored")).size(11.5));
        let why = if sampling { Some("a sampling pass is running — purge once it is done") } else { m.purge_refusal.as_deref() };
        let b = egui::Button::new(RichText::new(format!("Purge forgotten ({n})…")).color(theme::DANGER_INK));
        let r = super::super::probed("purge-forgotten", ui.add_enabled(why.is_none() && !st.purge_confirm, b)).on_hover_text(PURGE_TIP);
        let r = match why {
            Some(w) => r.on_disabled_hover_text(w),
            None => r,
        };
        if r.clicked() {
            st.purge_confirm = true;
        }
    });
    if st.purge_confirm {
        egui::Frame::new()
            .fill(RED.gamma_multiply(0.08))
            .stroke(egui::Stroke::new(1.0, RED.gamma_multiply(0.6)))
            .corner_radius(7)
            .inner_margin(egui::Margin::symmetric(9, 7))
            .show(ui, |ui| {
                ui.label(RichText::new(purge_confirm_text(n)).size(11.5));
                ui.horizontal(|ui| {
                    let purge = egui::Button::new(RichText::new("Purge").strong().color(Color32::WHITE)).fill(RED.gamma_multiply(0.8));
                    if super::super::probed("purge-confirm", ui.add(purge)).clicked() {
                        acts.push(Action::PurgeForgotten);
                        st.purge_confirm = false;
                    }
                    if super::super::probed("purge-cancel", ui.button("Cancel")).clicked() {
                        st.purge_confirm = false;
                    }
                });
            });
    }
    for s in &m.forgotten_samples {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(suspect_line(s)).monospace().size(11.5).color(theme::INK_DIM));
            if super::super::probed(&format!("restore-sample-{}", s.id.index), ui.small_button("Restore"))
                .on_hover_text("put this sample back into the fit (written at once; the model refits with it)")
                .clicked()
            {
                acts.push(Action::RestoreSample(s.id));
            }
        });
    }
}

/// "imported N, refused M" of an import (the Model tab).
pub fn import_line(r: &sim::ImportSummary) -> String {
    format!("last import: imported {}, refused {}", r.imported, r.refused.len())
}

/// The source line's hover text.
const SOURCE_LINE_TIP: &str = "where the estimated solver's model comes from: the shared collection of this solver surface (one folder in models/collections per h, σ, ionic potential, teacher and tolerance — every scene at those settings learns into it), a model file (a specialised per-scene model), or this scene only (a charged scene or a forced multiplicity keeps a per-scene model: collections are neutral until M9)";

fn warnbox(ui: &mut Ui, text: &str) {
    egui::Frame::new()
        .fill(AMBER.gamma_multiply(0.07))
        .stroke(egui::Stroke::new(1.0, AMBER.gamma_multiply(0.45)))
        .corner_radius(7)
        .inner_margin(egui::Margin::symmetric(9, 7))
        .show(ui, |ui| {
            ui.label(RichText::new(text).size(11.5).color(AMBER));
        });
}

/// Samples held but no model formed (`ModelStatus::no_model`; the 2026-10-20 PH₃ session): the
/// reason in plain words and, when it can help, **Retry** (re-solves a missing isolated-atom
/// reference, or refits — `Action::RetryFit`). Also in the Activity tab.
pub fn no_model_row(ui: &mut Ui, m: &ModelStatus, sampling: bool, acts: &mut Vec<Action>) {
    let Some(why) = &m.no_model else { return };
    warnbox(ui, why);
    if m.retry_fit || sampling {
        ui.horizontal_wrapped(|ui| {
            let b = egui::Button::new(RichText::new("Retry").strong().color(Color32::from_rgb(0x1c, 0x14, 0x05))).fill(AMBER);
            if super::super::probed("retry-fit", ui.add_enabled(!sampling && m.retry_fit, b))
                .on_disabled_hover_text("a pass is running")
                .on_hover_text("solve the missing isolated-atom reference — every allowed multiplicity, each from a cold start (a solve lost to a busy GPU is tried once more) — then fit the samples held (a short pass: pauses the dynamics). With every reference known it fits at once. Not offered when the atom's SCF itself failed: a cold solve is deterministic and would fail the same way.")
                .clicked()
            {
                acts.push(Action::RetryFit);
            }
            ui.weak("the samples are kept either way: Save model… writes them");
        });
    }
}

/// The running pass's progress bar and counts (also on the sampling card in the 3D view).
pub fn progress(ui: &mut Ui, p: &sim::SamplingProgress, m: Option<&ModelStatus>) {
    let frac = if p.total > 0 { p.done as f32 / p.total as f32 } else { 0.0 };
    let eta = m.and_then(|m| m.pass_eta_s).map_or(String::new(), |t| format!(" — about {} left", duration_label(t)));
    ui.add(egui::ProgressBar::new(frac).show_percentage().text(format!("{} — {}/{}{eta}", crate::ui::activity::words::pass_stage(p), p.done, p.total)));
    let e = if p.last_energy.is_finite() { format!("{:+.6} Ha", p.last_energy) } else { "—".into() };
    ui.weak(format!(
        "last teacher energy {e} · skipped {} · failed {} — the dynamics are paused; edits are allowed",
        p.skipped, p.failed
    ));
}

/// The long first-pass offer and its "Start anyway" (also on the offer card in the 3D view).
pub fn offer(ui: &mut Ui, o: &crate::sim::PassOffer, acts: &mut Vec<Action>) {
    ui.colored_label(
        AMBER,
        format!(
            "a first sampling pass here: ≈ {} at this scene's first solve ({} solves × {:.1} s; the panel's plan, {} solves, ≈ {}); a pass's own solves differ (references, warm starts) — the time left is shown once it runs. That is already the fewest solves that can earn trust on this scene. A smaller grid is faster.",
            duration_label(o.eta_s),
            o.solves,
            o.per_solve_s,
            o.full_solves,
            duration_label(o.full_eta_s)
        ),
    );
    if ui.button(RichText::new(format!("Start anyway (≈ {} at the first solve)", duration_label(o.eta_s))).color(AMBER)).clicked() {
        acts.push(Action::StartOffered(o.plan.clone()));
    }
}

/// The teach-around button's label, "Teach around N blockers (≈M solves, ≈T)", T at the model's
/// solve time; `None` when nothing is teachable (`ModelStatus::teach_offer`).
pub fn teach_label(m: &ModelStatus) -> Option<String> {
    let o = m.teach_offer.as_ref().filter(|o| !o.blockers.is_empty())?;
    let eta = m.solve_s.map(|s| format!(", ≈{}", super::super::activity::headline::clock(s * o.solves as f64))).unwrap_or_default();
    Some(format!("Teach around {} blocker{} (≈{} solves{eta})", o.blockers.len(), if o.blockers.len() == 1 { "" } else { "s" }, o.solves))
}

/// "Teach around N blockers (≈M solves, ≈T)" when an environment class is blocked by training
/// samples the rest of the data cannot predict (`ModelStatus::teach_offer`), with the blockers the
/// pass would not teach and why. The Model tab and the Activity tab's trust checklist both show it.
pub fn teach_row(ui: &mut Ui, m: &ModelStatus, sampling: bool, acts: &mut Vec<Action>) {
    let Some(o) = &m.teach_offer else { return };
    // nothing to say and nothing to teach (every blocker already refused, e.g. after two rounds):
    // no box — an empty pulsing outline read as a missing button (user report 2026-10-22); the
    // refusals below still say why
    let label = teach_label(m);
    if o.prompt.is_none() && label.is_none() {
        for (i, c, ins, why) in &o.refused {
            ui.weak(format!("#{i} ({c}, in-sample {ins:.2}×): not taught — {why}"));
        }
        if o.refused_total > o.refused.len() {
            ui.weak(format!("… and {} more not taught", o.refused_total - o.refused.len()));
        }
        return;
    }
    // the warning and its fix in one amber box whose outline pulses (user request 2026-10-19:
    // the button was easy to miss); still while a pass runs
    let pulse = if sampling {
        0.0
    } else {
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(40));
        (0.5 + 0.5 * (ui.input(|i| i.time) * 3.5).sin()) as f32
    };
    egui::Frame::new()
        .fill(AMBER.gamma_multiply(0.08))
        .stroke(egui::Stroke::new(2.0, AMBER.gamma_multiply(0.25 + 0.75 * pulse))) // fixed width: only the brightness pulses (a width change moved the layout)
        .corner_radius(8)
        .inner_margin(10)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            if let Some(p) = &o.prompt {
                ui.colored_label(AMBER, p);
            }
            if let Some(label) = &label {
                let names: Vec<String> = o.blockers.iter().map(|(i, c, loo, ins)| format!("#{i} ({c}: leave-one-out {loo:.1}×, in-sample {ins:.2}×)")).collect();
                ui.add_space(4.0);
                let button = egui::Button::new(RichText::new(format!("Fix it: {label}")).strong().size(14.0).color(egui::Color32::from_rgb(0x1c, 0x14, 0x05)))
                    .fill(AMBER)
                    .corner_radius(6)
                    .min_size(egui::Vec2::new(ui.available_width().min(360.0), 32.0));
                if ui
                    .add_enabled(!sampling, button)
                    .on_disabled_hover_text("a pass is running")
                    .on_hover_text(format!(
                        "a targeted pass around the training samples that keep a class untrusted: {}. About 10 DFT solves each — random nudges each clearly apart from the sample and from each other, and steps toward its nearest learned neighbour — of which 2 are held out to check the result. Accepted only if the unchanged class rule then passes with the sample still in the fit and the checks pass; at most two rounds. Pauses the dynamics.",
                        names.join(", ")
                    ))
                    .clicked()
                {
                    acts.push(Action::TeachAroundBlockers);
                }
            }
        });
    for (i, c, ins, why) in &o.refused {
        ui.weak(format!("#{i} ({c}, in-sample {ins:.2}×): not taught — {why}"));
    }
    if o.refused_total > o.refused.len() {
        ui.weak(format!("… and {} more not taught", o.refused_total - o.refused.len()));
    }
    if !o.blockers.is_empty() {
        for n in &o.notes {
            ui.weak(n);
        }
    }
}

/// The amber note on atoms with no neighbour within the model's cutoff (`ModelStatus::lone_note`):
/// they are not learned — every one of them, and the pass is refused. Decides nothing.
fn lone_note(ui: &mut Ui, input: &PanelInput<'_>, probe: &str) {
    if let Some(n) = input.model().and_then(|m| m.lone_note.as_deref()) {
        super::super::probed(probe, ui.label(RichText::new(n).color(AMBER)));
    }
}

/// Start sampling / Sample more (armed "Start anyway" above 2 min), the plan's cost, Cancel while a
/// pass runs.
fn sample_row(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    let sampling = input.sampling();
    // an empty scene has no species to sample (`sim::sampling::EMPTY_SCENE`)
    let empty = input.scene.nuclei.is_empty();
    let SampleButton { label, long, armed, cost } = sample_button(st, input);
    ui.horizontal_wrapped(|ui| {
        let text = if armed { RichText::new(label).color(AMBER) } else { RichText::new(label).strong() };
        let b = if armed { egui::Button::new(text) } else { egui::Button::new(text.color(Color32::from_rgb(0x17, 0x11, 0x2a))).fill(theme::ACCENT) };
        if super::super::probed("sample", ui.add_enabled(!sampling && !empty, b))
            .on_disabled_hover_text(if empty { super::super::EMPTY_SAMPLE } else { "a pass is running" })
            .on_hover_text("one pass over the current scene with the preset above and Plan settings below: isolated-atom references, random perturbations, pair scans, a hot MD burst; the samples are added to the model's and it is refitted. Pauses the dynamics. A pass longer than 2 min asks for a second click — estimated as solves × the last pass's measured rate, or × this scene's first solve before any pass (an estimate, not a bound: a pass's own solves differ; its time left is shown once it runs).")
            .clicked()
        {
            acts.push(Action::Sample { long });
        }
        if empty {
            ui.weak(super::super::EMPTY_SAMPLE);
        } else if let Some(c) = cost {
            ui.weak(c);
        }
        lone_note(ui, input, "lone-note");
        // Cancel only while a pass runs (the mockup; before, it was drawn greyed)
        if sampling
            && ui
            .add(egui::Button::new("Cancel"))
            .on_hover_text("end the pass now: what it sampled is kept; it is refitted only once there are enough samples to fit (Q′, the model's number of directions), else the Log says so and Sample more adds to them")
            .clicked()
        {
            acts.push(Action::CancelSampling);
        }
    });
}

/// Save model / Load model / Forget model (two clicks within 4 s): File ▸ Estimated model and
/// the Model tab.
/// Why audits matter, for the "audits on" switch's hover (user request 2026-10-18).
pub const AUDIT_WHY: &str = "Audits are the model's spot checks: every n-th answered step a real DFT solve is run at the same geometry and compared with the model's answer. They are how the app notices that the model has wandered somewhere it was never taught and is quietly answering wrong — a failed audit makes the model untrusted and hands the steps back to DFT, and the audit's answer becomes a training sample. Turning them off can make playback smoother (on the GPU the background audit competes with the display for the card, which can show as a stutter every so often), but nothing then checks the model while it answers: a wrong answer looks exactly like a right one. Keep them on unless you are only watching a geometry the model has seen many times.";

/// Shown under the cadence while audits are off.
pub const AUDIT_OFF_NOTE: &str = "Smoother, but blind: the model's answers are no longer checked, so errors outside what it learned go unnoticed. Turn audits back on before trusting what you see (the last cadence is restored).";

/// The native Save/Open dialog for model files, starting in `last`'s folder with its name
/// (user request 2026-10-18: pick the save location and the file to load); a bare name (nothing
/// chosen yet) starts in the dist folder's `models` (`scene::models_dir`) when there is one.
pub(in crate::ui) fn model_dialog(last: &std::path::Path) -> rfd::FileDialog {
    let mut d = rfd::FileDialog::new().add_filter("Atomic Sandbox model", &["model"]).add_filter("All files", &["*"]);
    let dir = last.parent().filter(|p| !p.as_os_str().is_empty()).map(std::path::Path::to_path_buf).or_else(crate::scene::models_dir).or_else(|| std::env::current_dir().ok());
    if let Some(dir) = dir {
        d = d.set_directory(dir);
    }
    if let Some(name) = last.file_name() {
        d = d.set_file_name(name.to_string_lossy());
    }
    d
}

pub fn model_file_buttons(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    expire_arming(st);
    let estimated = st.backend == Backend::Estimated;
    let m = input.model();
    let sampling = input.sampling();
    let fitted = m.is_some_and(|m| m.fitted);
    // samples without a fit are saved too (a samples-only file): a fit that did not form must not
    // lose the DFT taught so far
    let savable = fitted || m.is_some_and(|m| m.samples > 0);
    let path = st.model_file.clone().unwrap_or_else(|| model_path(&st.scene_path));
    ui.horizontal_wrapped(|ui| {
        let tip = if fitted {
            format!("choose where to write the model, its fit and its samples (last: {})", path.display())
        } else {
            format!("no model yet: choose where to write the samples taught so far — loading the file fits them once a fit can form (last: {})", path.display())
        };
        if ui
            .add_enabled(estimated && savable, egui::Button::new("Save model…"))
            .on_disabled_hover_text("nothing to save yet: no samples")
            .on_hover_text(tip)
            .clicked()
        {
            // the File window (`ui::files`): the models folder, with the native dialog one click away there
            super::super::files::open_save_model(st);
        }
        if ui
            .add_enabled(estimated && !sampling, egui::Button::new("Load model…"))
            .on_hover_text(
                "choose a .model file to read — it only runs on the grid, σ, charge, multiplicity, ionic potential, teacher and teacher tolerance it was trained at; if the scene differs, a card lists the differences and offers to apply the model's settings",
            )
            .clicked()
        {
            super::super::files::open(st);
        }
        let armed = forget_armed(st);
        let samples = m.map_or(0, |m| m.samples);
        // in collection mode only this session's own samples go: the collection's are permanent
        let in_collection = m.and_then(|m| match m.source {
            sim::ModelSource::Collection { samples: k, .. } => Some(k),
            _ => None,
        });
        let forget = match (armed, in_collection) {
            (true, Some(k)) => egui::Button::new(RichText::new(format!("Click again to forget {} session-only samples", samples.saturating_sub(k))).color(RED)),
            (true, None) => egui::Button::new(RichText::new(format!("Click again to forget {samples} samples")).color(RED)),
            (false, _) => egui::Button::new(RichText::new("Forget model…").color(theme::DANGER_INK)),
        };
        let tip = if in_collection.is_some() {
            "drop this session's own samples — the ones the shared collection did not take (two clicks within 4 s); the shared collection stays attached and keeps its samples: they are permanent, nothing forgets them"
        } else {
            "drop the model, its samples and references (two clicks within 4 s); the teacher answers every step until the next pass"
        };
        if super::super::probed("forget", ui.add_enabled(estimated && (m.is_some_and(|m| m.samples > 0 || m.fitted) || sampling), forget))
            .on_hover_text(tip)
            .clicked()
        {
            acts.push(Action::Forget);
        }
    });
}

/// The preset row (`ui/sampling_presets.rs`): picking one sets the plan's fields and keeps its
/// seed; the selection is read off the plan, so an edited field shows "Custom". The selected
/// preset's description heads the Plan settings fold.
fn preset_row(ui: &mut Ui, plan: &mut SamplingPlan) {
    let current = presets::matching(plan);
    ui.horizontal(|ui| {
        ui.label("preset").on_hover_text(presets::RULE_OF_THUMB);
        egui::ComboBox::from_id_salt("plan-preset")
            .selected_text(current.map_or(presets::CUSTOM, |i| presets::PRESETS[i].name))
            .show_ui(ui, |ui| {
                for (i, p) in presets::PRESETS.iter().enumerate() {
                    if ui.selectable_label(current == Some(i), p.name).on_hover_text(p.description).clicked() {
                        p.apply(plan);
                    }
                }
            })
            .response
            .on_hover_text(presets::RULE_OF_THUMB);
    });
}

/// The plan's fields (`sim::SamplingPlan`; the scan range stays at its default): the Plan
/// settings fold.
fn plan_fields(ui: &mut Ui, plan: &mut SamplingPlan, tu: super::super::TempUnits) {
    super::wide_grid(ui, "plan", 2).show(ui, |ui| {
        ui.label("perturbations");
        ui.add(egui::DragValue::new(&mut plan.perturbations).range(0..=2000).speed(1.0))
            .on_hover_text("random displacements of the scene, one teacher solve each");
        ui.end_row();
        ui.label("σ_r");
        ui.add(egui::DragValue::new(&mut plan.perturbation_sigma).range(0.0..=0.5).speed(0.002).fixed_decimals(3).suffix(" Bohr"))
            .on_hover_text("width of each coordinate's Gaussian displacement");
        ui.end_row();
        ui.label("hot burst");
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut plan.hot_burst_steps).range(0..=5000).speed(1.0).suffix(" steps"));
            // typed in the chosen scale, kept in kelvin (clamped at absolute zero)
            let mut shown = tu.from_k(plan.hot_burst_t);
            if ui.add(egui::DragValue::new(&mut shown).range(tu.range(0.0, 10000.0)).speed(10.0).max_decimals(2).suffix(tu.suffix())).changed() {
                plan.hot_burst_t = tu.to_k(shown);
            }
        })
        .response
        .on_hover_text("a short thermostatted DFT-MD run from the scene (dt 0.25 fs, CSVR τ 5 fs); every step is a sample");
        ui.end_row();
        ui.label("scan points / pair");
        ui.add(egui::DragValue::new(&mut plan.scan_points_per_pair).range(0..=200).speed(0.2)).on_hover_text(format!(
            "each pair closer than the model's cutoff is scanned from {}× to {}× its current separation",
            plan.scan_range.0, plan.scan_range.1
        ));
        ui.end_row();
        ui.label("seed");
        ui.add(egui::DragValue::new(&mut plan.seed).speed(0.2))
            .on_hover_text("advances after each pass, so a second pass draws new geometries");
        ui.end_row();
    });
}

/// The hover text of the "learned classes" rows.
pub const CLASS_LOO_HELP: &str = "after every refit each training sample is left out (exactly, in closed form) and the fit without it must predict it within the force bound and the energy tier — separately for \"pair\" environments (every atom with at most one neighbour within the model's cutoff) and \"many-body\" ones. The numbers are the worst left-out errors as multiples of those bounds (1× = at the bound). A class that fails, or holds fewer samples than the model directions its atoms can move along (per kind of atom and neighbours: the columns those environments touch, not the whole model's), is not answered by the model: its steps fall back to DFT. Until DFT has checked the model at three or more geometries of a class, one badly predicted sample is enough to keep the whole class on DFT; from then on the class is judged by the error those DFT checks measured (on average, with a margin to start answering), with the left-out average as a floor.";

/// One class's row: "pair: trusted · 273 samples · left-out force ≤ 0.10× · energy ≤ 0.15×".
/// One environment species' row under its class ("  many-body around O, with H: learned · 967
/// samples for 38 columns"): its samples against the columns its atoms touch (M8a user-fix 3), read
/// "12 of 146 samples needed" only when that count is what refuses it (the review's Minor 3).
pub fn species_line(class: &str, s: &sim::SpeciesVerdict) -> String {
    let count = if matches!(s.refusal, Some((sim::ClassKey::Samples, ..))) {
        format!("{} of {} samples needed", s.samples, s.columns)
    } else {
        format!("{} samples for {} columns", s.samples, s.columns)
    };
    format!("  {class} {}: {} · {count}", s.words(), if s.trusted { "learned" } else { "not learned yet (DFT)" })
}

pub fn class_line(c: &sim::ClassLoo) -> String {
    format!(
        "{}: {} · {} samples · left-out force ≤ {:.2}× · energy ≤ {:.2}×",
        c.class,
        if c.trusted { "trusted" } else { "not learned yet (DFT)" },
        c.samples,
        c.force_max,
        c.energy_max
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_class_row_names_the_class_its_trust_and_its_left_out_errors() {
        let c = sim::ClassLoo { class: "many-body", samples: 41, force_rms: 0.9, force_max: 2.69, energy_rms: 0.3, energy_max: 0.52, trusted: false, refusal: Some((sim::ClassKey::Force, 2.69, 1.0)), calibration: 1.0, measured: None, measured_energy: None, worst_atom: None, evidence: 0, trusted_since: None, inflation: 1.0, species: Vec::new() };
        assert_eq!(super::class_line(&c), "many-body: not learned yet (DFT) · 41 samples · left-out force ≤ 2.69× · energy ≤ 0.52×");
        let s = |z, nb: Vec<u8>, refusal: Option<(sim::ClassKey, f64, f64)>, samples, columns| sim::SpeciesVerdict { z, neighbours: nb, trusted: refusal.is_none(), refusal, calibration: 1.0, measured: None, measured_energy: None, worst_atom: None, evidence: 0, trusted_since: None, inflation: 1.0, samples, columns };
        assert_eq!(super::species_line("many-body", &s(8, vec![1], None, 967, 38)), "  many-body around O, with H: learned · 967 samples for 38 columns");
        assert_eq!(super::species_line("many-body", &s(1, vec![1, 8], Some((sim::ClassKey::Samples, 12.0, 146.0)), 12, 146)), "  many-body around H, with H and O: not learned yet (DFT) · 12 of 146 samples needed");
        assert_eq!(super::species_line("pair", &s(1, vec![7], Some((sim::ClassKey::Force, 2.5, 1.0)), 300, 29)), "  pair around H, with N: not learned yet (DFT) · 300 samples for 29 columns");
    }

    /// Spec 2026-11-12 R2: the "Worst-fitting samples" row text — the spec's own example, a ratio
    /// below 10 (one decimal), an undefined one (∞), and a forgotten sample the fit does not hold.
    #[test]
    fn a_worst_fitting_row_names_the_sample_and_its_left_out_ratio() {
        let s = |index, formula: &str, origin: &str, loo_ratio| sim::SampleSuspect { id: sim::SampleId { index, key: 0 }, formula: formula.into(), origin: origin.into(), loo_ratio };
        assert_eq!(super::suspect_line(&s(1804, "HO", "fallback", Some(38.46))), "#1804 · HO · fallback · 38× its bound");
        assert_eq!(super::suspect_line(&s(1133, "H2O", "pass perturbation", Some(5.23))), "#1133 · H2O · pass perturbation · 5.2× its bound");
        assert_eq!(super::suspect_line(&s(7, "H3", "pass scan", Some(f64::INFINITY))), "#7 · H3 · pass scan · ∞× its bound");
        assert_eq!(super::suspect_line(&s(40, "H2", "audit", None)), "#40 · H2 · audit · not in the fit");
    }
}

/// The Details fold's readout rows (the old grid, row for row): who answered this step, its
/// uncertainty against the threshold (a bar with the bound marked, and the numbers), trust (with
/// the reason when not), samples, epoch, the last fit, the strict verdict here, fit vs held, the
/// learned classes with their left-out ratios, the last refit, fallbacks over the last 100 steps
/// (bar), the teacher gap, the last audit and the booked force-mismatch work.
fn readouts(ui: &mut Ui, d: &sim::DynSnapshot, m: &ModelStatus, scene: &crate::scene::Scene, st: &UiState) {
    let mono = |t: String| RichText::new(t).monospace().size(11.5);
    row(ui, "this step", |ui| {
        ui.colored_label(source_colour(d.source), RichText::new(source_label(d.source)).strong()).on_hover_text(source_tip(d.source));
    });
    row(ui, "uncertainty", |ui| match d.uncertainty {
        Some(u) => {
            let c = if u <= m.force_bound { GREEN } else { AMBER };
            ui.colored_label(c, mono(format!("{u:.1e} / {:.1e} Ha/Bohr", m.force_bound))).on_hover_text(UNCERTAINTY_TIP);
        }
        None => {
            ui.colored_label(Color32::GRAY, "— (model not consulted)").on_hover_text(UNCERTAINTY_TIP);
        }
    });
    row(ui, "trusted", |ui| {
        let t = trust(Some(m), scene, st);
        let r = ui.colored_label(t.colour, t.text);
        // the plain line; then what it means, then the provider's own text
        let hover = [t.tip.map(str::to_string), t.detail.map(|d| format!("the fit's own words: {d}"))].into_iter().flatten().collect::<Vec<_>>().join("\n\n");
        if !hover.is_empty() {
            r.on_hover_text(hover);
        }
    });
    row(ui, "samples", |ui| ui.label(mono(format!("{}{}", m.samples, if m.pending_refit { " (refit pending)" } else { "" }))));
    row(ui, "model epoch", |ui| ui.label(mono(m.epoch.to_string())).on_hover_text("bumped whenever a refit moves the energy surface (the jump is booked into E conserved)"));
    row(ui, "last fit", |ui| match m.fit {
        Some(fit) => ui.label(mono(last_fit_line(fit, m.body4_degree))),
        None => ui.colored_label(Color32::GRAY, "—"),
    });
    // the training samples the fit cannot reproduce (the misfit set M, spec 2026-11-13 §4.3)
    if let Some((c, line)) = misfit(m) {
        row(ui, "misfit", |ui| ui.colored_label(if c.met { theme::INK } else { AMBER }, mono(line)).on_hover_text(MISFIT_TIP));
    }
    if let Some(r) = &m.readiness {
        row(ui, "strict here", |ui| {
            let text = if r.answers { "answers".to_string() } else { format!("refuses — {}", r.text) };
            ui.colored_label(if r.answers { GREEN } else { AMBER }, text).on_hover_text("the fallback policy's verdict at this geometry with draft mode off: whether the model would answer a step here, else the rule that sends it to DFT");
        });
        if let Some(u) = &r.unlearned {
            row(ui, "this scene", |ui| ui.colored_label(AMBER, u));
        }
    }
    if let Some(text) = &m.fit_vs_held {
        row(ui, "fit vs held", |ui| ui.label(mono(text.clone())).on_hover_text(FIT_VS_HELD_HELP));
    }
    if !m.class_loo.is_empty() {
        // final review app M4: the class-wise trust behind "not learned well enough yet"
        row(ui, "learned classes", |ui| {
            for c in &m.class_loo {
                ui.colored_label(if c.trusted { GREEN } else { AMBER }, mono(class_line(c))).on_hover_text(CLASS_LOO_HELP);
                // per environment species (M8a user-test fix 2): an atom answers by its own
                // … and always under a refused class, so the short key is named (M8a user-fix 3)
                for s in c.species.iter().filter(|_| c.species.len() > 1 || !c.trusted) {
                    ui.colored_label(if s.trusted { GREEN } else { AMBER }, mono(species_line(c.class, s))).on_hover_text(CLASS_LOO_HELP);
                }
            }
        });
    }
    if let Some(r) = &m.last_refit {
        row(ui, "last refit", |ui| match &r.result {
            Ok(n) => ui.label(mono(format!("{n} samples, {:.0} ms, at step {} ({} this run)", r.ms, r.at_step, m.refits))),
            Err(e) => ui.colored_label(AMBER, format!("failed at step {}: {e}", r.at_step)),
        });
    }
    row(ui, "fallbacks", |ui| {
        let f = d.fallback_fraction_100 as f32;
        let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 6.0), Sense::hover());
        ui.painter().rect_filled(rect, 3.0, theme::PANEL_3);
        let mut fill = rect;
        fill.set_width(rect.width() * f.clamp(0.0, 1.0));
        ui.painter().rect_filled(fill, 3.0, AMBER);
        ui.label(mono(format!("{:.0} % of the last 100 steps", 100.0 * d.fallback_fraction_100)));
    });
    row(ui, "teacher gap", |ui| {
        match (d.teacher_gap, m.last_gap) {
            (Some(g), _) => ui.label(mono(format!("{g:+.2e} Ha (this step)"))),
            (None, Some((g, step))) => ui.label(mono(format!("{g:+.2e} Ha at step {step}"))),
            (None, None) => ui.colored_label(Color32::GRAY, "— (no fallback on the model's surface yet)"),
        }
        .on_hover_text("E_DFT − E_model on the most recent fallback or audit booked on the model's surface (the energy stays on the model's surface; the gap is the model's error there)");
    });
    row(ui, "last audit", |ui| {
        let (c, full, _) = audit_words(m);
        ui.colored_label(c, full).on_hover_text(audit_tip(m));
    });
    row(ui, "mismatch work", |ui| {
        ui.label(mono(format!("{:+.2e} Ha", d.work_mismatch)))
            .on_hover_text("work of (F_DFT − F_model) on fallback steps, booked into E conserved so the drift measures the integrator, not the model");
    });
}

/// The uncertainty against its bound: the bound sits at the middle of the bar, the value's dot
/// on a scale to twice the bound.
fn uncertainty_bar(ui: &mut Ui, u: f64, bound: f64, c: Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 10.0), egui::Sense::hover());
    let p = ui.painter();
    let mid = rect.left() + rect.width() / 2.0;
    let left = egui::Rect::from_min_max(rect.left_top(), egui::pos2(mid, rect.bottom()));
    let right = egui::Rect::from_min_max(egui::pos2(mid, rect.top()), rect.right_bottom());
    p.rect_filled(left, 4.0, GREEN.gamma_multiply(0.25));
    p.rect_filled(right, 4.0, AMBER.gamma_multiply(0.22));
    p.line_segment([egui::pos2(mid, rect.top() - 2.0), egui::pos2(mid, rect.bottom() + 2.0)], egui::Stroke::new(2.0, theme::INK));
    let x = if bound > 0.0 { (u / (2.0 * bound)).clamp(0.0, 1.0) as f32 } else { 1.0 };
    p.circle_filled(egui::pos2(rect.left() + x * rect.width(), rect.center().y), 5.0, c);
}

/// The interrupted training's offer (`UiState::resume`; user report 2026-10-20): resume the pass
/// where it stopped, on its own scene and in the estimated solver, or discard it. Shown on every
/// solver; hidden while a pass runs (a pass must end before another resumes).
fn resume_row(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    if input.sampling() {
        // a running pass owns the folder now (a resumed one, or a new one replacing it)
        st.resume = None;
        return;
    }
    let Some(o) = st.resume.as_ref() else { return };
    let z: Vec<u8> = o.scene.nuclei.iter().map(|n| n.z).collect();
    let what = crate::names::formula(&z, (o.scene.charge != 0).then_some(o.scene.charge));
    egui::Frame::new().fill(AMBER.gamma_multiply(0.08)).stroke(egui::Stroke::new(1.5, AMBER.gamma_multiply(0.6))).inner_margin(6.0).corner_radius(4.0).show(ui, |ui| {
        ui.colored_label(
            AMBER,
            format!(
                "An interrupted training: {what} at {}³ × {} Bohr, sample {} of {} ({}). Its samples are on disk.",
                o.scene.grid_n,
                o.scene.grid_h,
                (o.done + 1).min(o.total),
                o.total,
                o.stage
            ),
        );
        ui.horizontal(|ui| {
            let resume = ui
                .button(RichText::new(format!("Resume the interrupted training (sample {} of {})", (o.done + 1).min(o.total), o.total)).color(AMBER))
                .on_hover_text("switches to the estimated solver on the pass's own scene (its grid, σ, charge and ionic potential), loads the samples it had taught and goes on with its remaining jobs — the current scene and model are replaced");
            if resume.clicked() {
                acts.push(Action::ResumeTraining);
            }
            if ui.button("Discard").on_hover_text("removes the interrupted pass's files; a new sampling pass also replaces them").clicked() {
                acts.push(Action::DiscardTraining);
            }
        });
    });
}
