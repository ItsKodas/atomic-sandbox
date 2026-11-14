//! The built-in layouts (docking spec D-11, D2; user decision 2026-11-03: layouts live in the
//! profile, four built-ins ship, and choosing one rearranges the current profile, which then
//! remembers it — the profile's autosave captures the tree and the hidden list like any other
//! rearrangement). Window ▸ Layouts lists them (spec D-7), Reset layout applies One monitor (Two
//! monitors when a second monitor is connected), and
//! the screenshot harness's `--ui layout=<one|two|training|presenting>` (and `dock=closed`, which
//! is Presenting) poses them (spec D-13). Display state only: nothing here is an `Action`.
//!
//! The arrangements, as the profile writes them (`layout = …` / `hidden = …`):
//!
//! - **One monitor** — the D1 default ([`layout::default_layout`]), nothing hidden.
//! - **Two monitors** — the default minus Activity and Readouts (the Graph panel merged into it on
//!   2026-11-14), `H(0.24,V(0.55,[Model*,Knowledge],[Selection*]),H(0.70,3D,V(0.62,[View*,Orbitals],
//!   [Log*])))`, and Activity over Readouts, `V(0.50,[Activity*],[Readouts*])`, in a detached window filling the second monitor's work area
//!   (maximised there, not full screen: [`Preset::window`], D3, `super::detached`). With one monitor
//!   (or none known, headless) it is One monitor and the status says "Only one monitor found"
//!   (`UiState::apply_layout`).
//! - **Training** — `H(0.34,[Model*,Knowledge],H(0.55,3D,V(0.50,[Activity*],[Readouts*])))`: Model /
//!   Knowledge on the left at 34 % of the window, Activity over Readouts (its E total / T chart) on
//!   the right at 30 % of the window (0.45 of the 66 % the left column leaves; controller ruling,
//!   Task 7 review: at 30 % of that 66 % the old Graph's controls were clipped), the 3D view ≈ 36 %
//!   between them; View / Orbitals / Selection / Log auto-hidden on the right
//!   edge at [`TRAINING_HIDDEN`] points (recorded for review).
//! - **Presenting** — the tree is just `3D`; every panel is auto-hidden, the default layout's left
//!   column (Model, Knowledge, Selection, Readouts) on the left edge at [`LEFT_SIZE`], the rest on
//!   the right at [`RIGHT_SIZE`] — the widths those columns have in the default layout at the
//!   window's default 1400 pt (recorded for review: each panel slides out on the side it lives on).
use super::autohide;
use super::layout::{self, Edge, Hidden, Node};
use crate::ui::DockTab;

/// A built-in layout (spec D-11).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    OneMonitor,
    TwoMonitors,
    Training,
    Presenting,
}

/// Training's auto-hidden panels' flyout width (points): wide enough for the Readouts' tiles and
/// the Log's lines.
pub const TRAINING_HIDDEN: f32 = 340.0;
/// Presenting's hidden widths: the default left column's (24 % of 1400 pt) and right column's
/// (≈ 30 % of the 76 % it leaves: [`autohide::DEFAULT_SIZE`]).
pub const LEFT_SIZE: f32 = 336.0;
pub const RIGHT_SIZE: f32 = autohide::DEFAULT_SIZE;

impl Preset {
    pub const ALL: [Preset; 4] = [Preset::OneMonitor, Preset::TwoMonitors, Preset::Training, Preset::Presenting];

    /// Its Window ▸ Layouts row.
    pub fn title(self) -> &'static str {
        match self {
            Preset::OneMonitor => "One monitor",
            Preset::TwoMonitors => "Two monitors",
            Preset::Training => "Training",
            Preset::Presenting => "Presenting",
        }
    }

    /// The screenshot harness's word (`--ui layout=<word>`), also its menu row's probe.
    pub fn word(self) -> &'static str {
        match self {
            Preset::OneMonitor => "one",
            Preset::TwoMonitors => "two",
            Preset::Training => "training",
            Preset::Presenting => "presenting",
        }
    }

    /// The preset a word names: its [`Preset::word`] or [`Preset::title`], any case.
    pub fn from_word(w: &str) -> Option<Preset> {
        let w = w.trim();
        Preset::ALL.into_iter().find(|p| w.eq_ignore_ascii_case(p.word()) || w.eq_ignore_ascii_case(p.title()))
    }

    /// Its menu row's tooltip.
    pub fn tip(self) -> &'static str {
        match self {
            Preset::OneMonitor => "the default: Model / Knowledge over Selection / Readouts on the left, the 3D view, View / Orbitals over Log / Activity on the right",
            Preset::TwoMonitors => "the default, with Activity over Readouts in their own window filling the second monitor (with one monitor: One monitor)",
            Preset::Training => "for teaching the estimated solver: Model / Knowledge on the left, Activity over Readouts on the right; View, Orbitals, Selection and Log fold to the right edge (hover their icons)",
            Preset::Presenting => "the 3D view fills the window: every panel folds to an icon on the view's left or right edge (hover one to slide it out)",
        }
    }

    /// The arrangement: the tree and the auto-hidden panels (see the module doc).
    pub fn tree(self) -> (Node, Vec<Hidden>) {
        use DockTab::*;
        match self {
            Preset::OneMonitor => (layout::default_layout(), Vec::new()),
            Preset::TwoMonitors => {
                let main = split(
                    true,
                    0.24,
                    split(false, 0.55, leaf(&[Model, Knowledge]), leaf(&[Selection])),
                    split(true, 0.70, Node::View3d, split(false, 0.62, leaf(&[View, Orbitals]), leaf(&[Log]))),
                );
                (main, Vec::new())
            }
            Preset::Training => {
                let tree = split(true, 0.34, leaf(&[Model, Knowledge]), split(true, 0.55, Node::View3d, split(false, 0.50, leaf(&[Activity]), leaf(&[Readouts]))));
                let hidden = [View, Orbitals, Selection, Log].map(|tab| Hidden { tab, edge: Edge::Right, size: TRAINING_HIDDEN });
                (tree, hidden.to_vec())
            }
            Preset::Presenting => {
                // each panel to the side it lives on in the default layout (its root split's
                // first child is the left column)
                let d = layout::default_layout();
                let left = match &d {
                    Node::Split { a, .. } => layout::panels(a),
                    _ => Vec::new(),
                };
                let (l, r): (Vec<DockTab>, Vec<DockTab>) = layout::panels(&d).into_iter().partition(|t| left.contains(t));
                let fold = |tabs: Vec<DockTab>, edge, size| tabs.into_iter().map(move |tab| Hidden { tab, edge, size });
                (Node::View3d, fold(l, Edge::Left, LEFT_SIZE).chain(fold(r, Edge::Right, RIGHT_SIZE)).collect())
            }
        }
    }
}

impl Preset {
    /// The detached window's tree (spec D-11, D-12): Two monitors' Activity over Readouts, for the
    /// second monitor; the others have none.
    pub fn window(self) -> Option<Node> {
        use DockTab::*;
        (self == Preset::TwoMonitors).then(|| split(false, 0.50, leaf(&[Activity]), leaf(&[Readouts])))
    }
}

fn leaf(tabs: &[DockTab]) -> Node {
    Node::Leaf { tabs: tabs.to_vec(), active: 0 }
}

fn split(horizontal: bool, frac: f32, a: Node, b: Node) -> Node {
    Node::Split { horizontal, frac, a: Box::new(a), b: Box::new(b) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::workspace::Workspace;

    #[test]
    fn every_preset_holds_every_panel_exactly_once() {
        for p in Preset::ALL {
            let (tree, hidden) = p.tree();
            assert!(layout::has_view(&tree), "{p:?}");
            let mut all = layout::panels(&tree);
            all.extend(hidden.iter().map(|h| h.tab));
            all.extend(p.window().map(|w| layout::panels(&w)).unwrap_or_default());
            for t in DockTab::PANELS {
                assert_eq!(all.iter().filter(|&&x| x == t).count(), 1, "{p:?}: {t:?} once in {} + {}", layout::encode(&tree), layout::encode_hidden(&hidden));
            }
            assert_eq!(all.len(), DockTab::PANELS.len(), "{p:?}");
            assert!(hidden.iter().all(|h| (layout::HIDDEN_MIN..=layout::HIDDEN_MAX).contains(&h.size)), "{p:?}: sizes in bounds");
        }
    }

    #[test]
    fn the_presets_are_the_spec_arrangements() {
        let (one, h) = Preset::OneMonitor.tree();
        assert_eq!((one.clone(), h.is_empty()), (layout::default_layout(), true));
        let (two, h) = Preset::TwoMonitors.tree();
        assert_eq!((layout::encode(&two), h.is_empty()), ("H(0.24,V(0.55,[Model*,Knowledge],[Selection*]),H(0.70,3D,V(0.62,[View*,Orbitals],[Log*])))".to_string(), true));
        assert_eq!(layout::encode(&Preset::TwoMonitors.window().unwrap()), "V(0.50,[Activity*],[Readouts*])");
        assert!(Preset::ALL.iter().filter(|p| **p != Preset::TwoMonitors).all(|p| p.window().is_none()));
        let _ = one;
        let (tree, hidden) = Preset::Training.tree();
        assert_eq!(layout::encode(&tree), "H(0.34,[Model*,Knowledge],H(0.55,3D,V(0.50,[Activity*],[Readouts*])))");
        assert_eq!(layout::encode_hidden(&hidden), "View:right:340, Orbitals:right:340, Selection:right:340, Log:right:340");
        let (tree, hidden) = Preset::Presenting.tree();
        assert_eq!(tree, Node::View3d, "Presenting's tree is just the 3D view");
        assert_eq!(
            layout::encode_hidden(&hidden),
            "Model:left:336, Knowledge:left:336, Selection:left:336, Readouts:left:336, View:right:320, Orbitals:right:320, Log:right:320, Activity:right:320"
        );
    }

    /// Applied to a workspace, then encoded as the profile writes it and parsed back: the same
    /// arrangement — what the profile's autosave keeps is the preset.
    #[test]
    fn applying_encoding_and_parsing_returns_the_same_arrangement() {
        for p in Preset::ALL {
            let (tree, hidden) = p.tree();
            let mut w = Workspace::default();
            w.set_node(layout::parse("H(0.5,[Log*,Readouts],V(0.3,3D,[Model*]))").unwrap());
            w.pin(DockTab::Orbitals, false);
            w.set_node(tree.clone());
            w.set_hidden(hidden.clone());
            assert_eq!(w.to_node(), tree, "{p:?}");
            assert_eq!(w.hidden(), &hidden[..], "{p:?}");
            let (text, htext) = (layout::encode(&w.to_node()), layout::encode_hidden(w.hidden()));
            assert_eq!(layout::parse(&text), Some(tree.clone()), "{p:?}: {text}");
            assert_eq!(layout::parse_hidden(&htext), hidden, "{p:?}: {htext}");
        }
    }

    #[test]
    fn words_and_titles_name_the_presets() {
        for p in Preset::ALL {
            assert_eq!(Preset::from_word(p.word()), Some(p));
            assert_eq!(Preset::from_word(&p.title().to_uppercase()), Some(p));
        }
        assert_eq!(Preset::from_word(" Training "), Some(Preset::Training));
        assert_eq!(Preset::from_word("three"), None);
        assert_eq!(Preset::ALL.map(Preset::title), ["One monitor", "Two monitors", "Training", "Presenting"]);
    }
}
