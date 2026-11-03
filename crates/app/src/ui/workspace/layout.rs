//! The workspace's layout tree and its text form (docking spec D-8, D-9). The profile saves the
//! tree as one `layout = <tree>` line in our own small grammar rather than the dock library's serde
//! (user decision, spec D-8): the profile is hand-readable `key = value` text and must survive a
//! library upgrade. The grammar (spec D-9):
//!
//! ```text
//! node  := "3D" | leaf | split
//! leaf  := "[" tab ("," tab)* "]"        tab := Name ["*"]   (* = the active tab)
//! split := ("H" | "V") "(" frac "," node "," node ")"   frac = the first child's share, 0.05..0.95
//! ```
//!
//! `H` puts its children side by side, `V` stacks them. A tab is written as its one-word
//! [`DockTab::name`] and read case-insensitively by that name or its [`DockTab::title`];
//! whitespace between tokens is ignored. Reading is total over the tree's content ([`clean`]): an
//! unknown name is dropped, a panel (or the 3D view) met twice keeps its first place (depth first,
//! the first child before the second), an empty leaf goes and its split collapses into the other
//! child, `frac` is clamped. Only text that is not the grammar at all, nesting deeper than
//! [`MAX_DEPTH`], or a tree left without the 3D view does not parse — the caller then uses
//! [`default_layout`]. [`encode`] writes `frac` with two decimals (1 % of the window, which is
//! what a saved layout keeps) and always stars the active tab.
//!
//! The auto-hidden panels (spec D-8, D-10; D2) are a second line, `hidden = Graph:left:340,
//! Log:right:300`: each panel's name, the 3D view's edge it folded to and its remembered size in
//! points (its width; its height on the bottom edge). [`parse_hidden`] is total too: names and
//! edges are read in any case, a size is clamped to [`HIDDEN_MIN`]..=[`HIDDEN_MAX`], an entry
//! whose name, edge or size cannot be read is dropped, and a panel listed twice keeps its first
//! entry.

use crate::ui::DockTab;
use std::fmt::Write as _;

/// The bounds of a split's `frac`: neither child shrinks below 5 % of its parent.
const FRAC_MIN: f32 = 0.05;
const FRAC_MAX: f32 = 0.95;
/// The deepest nesting [`parse`] reads (a damaged profile must not overflow the stack; a real
/// layout is a handful of levels deep).
pub const MAX_DEPTH: usize = 64;

/// One node of the workspace tree.
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    /// The 3D view: a tile, not a panel (spec D-3) — exactly one per main-window tree.
    View3d,
    /// Panels tabbed together; `active` indexes the one shown (out of range reads as the first).
    Leaf { tabs: Vec<DockTab>, active: usize },
    /// Two children: side by side when `horizontal` (`H`), else stacked (`V`); `frac` is the first
    /// child's share of the parent, [`FRAC_MIN`]..=[`FRAC_MAX`].
    Split { horizontal: bool, frac: f32, a: Box<Node>, b: Box<Node> },
}

fn leaf(tabs: Vec<DockTab>, active: usize) -> Node {
    Node::Leaf { tabs, active }
}

fn split(horizontal: bool, frac: f32, a: Node, b: Node) -> Node {
    Node::Split { horizontal, frac, a: Box::new(a), b: Box::new(b) }
}

/// Reads a saved layout and [`clean`]s it. `None` when the text is not the grammar (or nests
/// deeper than [`MAX_DEPTH`]), or when no 3D view is left after cleaning: the caller falls back to
/// [`default_layout`].
pub fn parse(s: &str) -> Option<Node> {
    let mut r = Reader { s, at: 0 };
    let n = r.node(0)?;
    if r.peek().is_some() {
        return None; // trailing text
    }
    clean(n).filter(has_view)
}

/// The tree as text, in the grammar [`parse`] reads.
pub fn encode(n: &Node) -> String {
    let mut out = String::new();
    encode_into(n, &mut out);
    out
}

fn encode_into(n: &Node, out: &mut String) {
    match n {
        Node::View3d => out.push_str("3D"),
        Node::Leaf { tabs, active } => {
            let active = if *active < tabs.len() { *active } else { 0 };
            out.push('[');
            for (i, t) in tabs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(t.name());
                if i == active {
                    out.push('*');
                }
            }
            out.push(']');
        }
        Node::Split { horizontal, frac, a, b } => {
            let _ = write!(out, "{}({frac:.2},", if *horizontal { 'H' } else { 'V' });
            encode_into(a, out);
            out.push(',');
            encode_into(b, out);
            out.push(')');
        }
    }
}

/// The D1 default (spec D-9's example, exactly): Model / Knowledge over Selection / Readouts on the
/// left (24 %), the 3D view, View / Orbitals over Log / Activity / Graph on the right (30 %).
/// Activity and Graph sat in the lower-left leaf until docking Task 4 (2026-11-03): with four tabs
/// there the Graph tab was scrolled off its bar at 1400 pt.
pub fn default_layout() -> Node {
    use DockTab::*;
    split(
        true,
        0.24,
        split(false, 0.55, leaf(vec![Model, Knowledge], 0), leaf(vec![Selection, Readouts], 0)),
        split(true, 0.70, Node::View3d, split(false, 0.62, leaf(vec![View, Orbitals], 0), leaf(vec![Log, Activity, Graph], 0))),
    )
}

/// The old profile's migration (spec D-8): its `left_tabs` / `right_tabs` lists into the D1
/// default's shape and fractions. The mapping, deterministic:
///
/// - a panel listed twice keeps its first place (the left list is read before the right);
/// - the left column (24 %) is one leaf of the left list in order, omitted when that is empty;
/// - the right column (30 %, beside the 3D view) is `V(0.62, top, [Log])` when the right list
///   holds Log, else one leaf `top`; `top` is the right list in order without that Log;
/// - View, unless a list already holds it, joins `top` as its second tab (its first when `top`
///   is otherwise empty);
/// - any other panel neither list names joins the right column's last leaf, in
///   [`DockTab::PANELS`] order;
/// - each leaf's first tab is active; two empty lists give [`default_layout`].
pub fn from_sides(left: &[DockTab], right: &[DockTab]) -> Node {
    if left.is_empty() && right.is_empty() {
        return default_layout();
    }
    let mut seen = Vec::new();
    let mut firsts = |list: &[DockTab]| -> Vec<DockTab> {
        list.iter()
            .copied()
            .filter(|t| {
                let new = !seen.contains(t);
                if new {
                    seen.push(*t);
                }
                new
            })
            .collect()
    };
    let left = firsts(left);
    let right = firsts(right);
    let log_right = right.contains(&DockTab::Log);
    let mut top: Vec<DockTab> = right.into_iter().filter(|&t| !(log_right && t == DockTab::Log)).collect();
    if !seen.contains(&DockTab::View) {
        top.insert(top.len().min(1), DockTab::View);
        seen.push(DockTab::View);
    }
    let missing = DockTab::PANELS.into_iter().filter(|t| !seen.contains(t));
    let mut bottom = if log_right { vec![DockTab::Log] } else { Vec::new() };
    if log_right {
        bottom.extend(missing);
    } else {
        top.extend(missing);
    }
    let right_col = match (top.is_empty(), bottom.is_empty()) {
        (false, false) => Some(split(false, 0.62, leaf(top, 0), leaf(bottom, 0))),
        (false, true) => Some(leaf(top, 0)),
        (true, false) => Some(leaf(bottom, 0)),
        (true, true) => None,
    };
    let centre = match right_col {
        Some(r) => split(true, 0.70, Node::View3d, r),
        None => Node::View3d,
    };
    if left.is_empty() { centre } else { split(true, 0.24, leaf(left, 0), centre) }
}

/// Makes a tree well formed: a panel or 3D view met a second time is dropped (depth first, so the
/// first place wins), the active tab stays the same panel when others go (else the first), an
/// empty leaf is removed and its split collapses into the other child, `frac` is clamped to
/// [`FRAC_MIN`]..=[`FRAC_MAX`] (NaN takes the middle). `None` when nothing is left. It does not
/// require a 3D view (a detached window's tree has none, spec D-12); [`parse`] does.
pub fn clean(n: Node) -> Option<Node> {
    clean_in(n, &mut Vec::new(), &mut false)
}

fn clean_in(n: Node, seen: &mut Vec<DockTab>, view_seen: &mut bool) -> Option<Node> {
    match n {
        Node::View3d => (!std::mem::replace(view_seen, true)).then_some(Node::View3d),
        Node::Leaf { tabs, active } => {
            let shown = tabs.get(active).copied();
            let mut kept = Vec::with_capacity(tabs.len());
            for t in tabs {
                if !seen.contains(&t) {
                    seen.push(t);
                    kept.push(t);
                }
            }
            let active = shown.and_then(|s| kept.iter().position(|&t| t == s)).unwrap_or(0);
            (!kept.is_empty()).then_some(Node::Leaf { tabs: kept, active })
        }
        Node::Split { horizontal, frac, a, b } => {
            let frac = if frac.is_nan() { 0.5 } else { frac.clamp(FRAC_MIN, FRAC_MAX) };
            match (clean_in(*a, seen, view_seen), clean_in(*b, seen, view_seen)) {
                (Some(a), Some(b)) => Some(split(horizontal, frac, a, b)),
                (Some(only), None) | (None, Some(only)) => Some(only),
                (None, None) => None,
            }
        }
    }
}

/// The tree's panels in reading order (depth first, each leaf's tabs in order); not the 3D view.
pub fn panels(n: &Node) -> Vec<DockTab> {
    fn walk(n: &Node, out: &mut Vec<DockTab>) {
        match n {
            Node::View3d => {}
            Node::Leaf { tabs, .. } => out.extend_from_slice(tabs),
            Node::Split { a, b, .. } => {
                walk(a, out);
                walk(b, out);
            }
        }
    }
    let mut out = Vec::new();
    walk(n, &mut out);
    out
}

/// Whether the tree holds the 3D view (a main-window tree must, spec D-3).
pub fn has_view(n: &Node) -> bool {
    match n {
        Node::View3d => true,
        Node::Leaf { .. } => false,
        Node::Split { a, b, .. } => has_view(a) || has_view(b),
    }
}

/// The edge of the 3D view an auto-hidden panel folds to (spec D-10): its strip and its flyout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Edge {
    Left,
    Right,
    Bottom,
}

impl Edge {
    pub const ALL: [Edge; 3] = [Edge::Left, Edge::Right, Edge::Bottom];

    /// The profile's word for the edge.
    pub fn word(self) -> &'static str {
        match self {
            Edge::Left => "left",
            Edge::Right => "right",
            Edge::Bottom => "bottom",
        }
    }
}

/// An auto-hidden panel (spec D-10): the edge it folded to and its remembered size in points —
/// its leaf's width, or height on the bottom edge — which its flyout opens at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hidden {
    pub tab: DockTab,
    pub edge: Edge,
    pub size: f32,
}

/// The bounds of a hidden panel's remembered size (points).
pub const HIDDEN_MIN: f32 = 160.0;
pub const HIDDEN_MAX: f32 = 900.0;

/// The `hidden = …` line's value: `Name:edge:size` entries joined by `, `, sizes in whole points.
pub fn encode_hidden(h: &[Hidden]) -> String {
    h.iter().map(|h| format!("{}:{}:{:.0}", h.tab.name(), h.edge.word(), h.size)).collect::<Vec<_>>().join(", ")
}

/// Reads a `hidden = …` value (see the module doc: total, sizes clamped, the unreadable dropped,
/// a duplicate keeps its first entry).
pub fn parse_hidden(s: &str) -> Vec<Hidden> {
    let mut out: Vec<Hidden> = Vec::new();
    for entry in s.split(',') {
        let mut parts = entry.split(':').map(str::trim);
        let (Some(name), Some(edge), Some(size), None) = (parts.next(), parts.next(), parts.next(), parts.next()) else { continue };
        let Some(tab) = tab_named(name) else { continue };
        let Some(edge) = Edge::ALL.into_iter().find(|e| edge.eq_ignore_ascii_case(e.word())) else { continue };
        let Some(size) = size.parse::<f32>().ok().filter(|x| !x.is_nan()) else { continue };
        if !out.iter().any(|h| h.tab == tab) {
            out.push(Hidden { tab, edge, size: size.clamp(HIDDEN_MIN, HIDDEN_MAX) });
        }
    }
    out
}

/// The panel a saved name means: its [`DockTab::name`] or [`DockTab::title`], any case.
fn tab_named(w: &str) -> Option<DockTab> {
    DockTab::PANELS.into_iter().find(|t| w.eq_ignore_ascii_case(t.name()) || w.eq_ignore_ascii_case(t.title()))
}

/// A cursor over the layout text. The grammar's punctuation is ASCII, so scanning bytes for it
/// only ever stops on a char boundary (a title's "·" is passed over whole).
struct Reader<'a> {
    s: &'a str,
    at: usize,
}

impl<'a> Reader<'a> {
    /// The next byte after any whitespace (not consumed).
    fn peek(&mut self) -> Option<u8> {
        let b = self.s.as_bytes();
        while b.get(self.at).is_some_and(|c| c.is_ascii_whitespace()) {
            self.at += 1;
        }
        b.get(self.at).copied()
    }

    /// Consumes `c` if it is next.
    fn eat(&mut self, c: u8) -> Option<()> {
        (self.peek()? == c).then(|| self.at += 1)
    }

    /// The text up to the next grammar punctuation (not consumed), trimmed.
    fn word(&mut self) -> &'a str {
        let start = self.at;
        while self.s.as_bytes().get(self.at).is_some_and(|c| !b"[](),*".contains(c)) {
            self.at += 1;
        }
        self.s[start..self.at].trim()
    }

    fn node(&mut self, depth: usize) -> Option<Node> {
        if depth > MAX_DEPTH {
            return None;
        }
        if self.peek()? == b'[' {
            return self.leaf();
        }
        let w = self.word();
        if w.eq_ignore_ascii_case("3D") {
            return Some(Node::View3d);
        }
        let horizontal = match w {
            "H" | "h" => true,
            "V" | "v" => false,
            _ => return None,
        };
        self.eat(b'(')?;
        let frac: f32 = self.word().parse().ok()?;
        self.eat(b',')?;
        let a = self.node(depth + 1)?;
        self.eat(b',')?;
        let b = self.node(depth + 1)?;
        self.eat(b')')?;
        Some(split(horizontal, frac, a, b))
    }

    /// `[` tab (`,` tab)* `]`, also `[]`; unknown names are skipped here, duplicates by [`clean`].
    fn leaf(&mut self) -> Option<Node> {
        self.eat(b'[')?;
        let (mut tabs, mut active) = (Vec::new(), None);
        if self.eat(b']').is_some() {
            return Some(leaf(tabs, 0));
        }
        loop {
            let name = self.word();
            let starred = self.eat(b'*').is_some();
            if let Some(t) = tab_named(name) {
                if starred && active.is_none() {
                    active = Some(tabs.len());
                }
                tabs.push(t);
            }
            match self.peek()? {
                b',' => self.at += 1,
                b']' => {
                    self.at += 1;
                    return Some(leaf(tabs, active.unwrap_or(0)));
                }
                _ => return None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use DockTab::*;

    /// The old docks' default tab lists (before docking Task 2), as an old profile holds them.
    const OLD_LEFT: [DockTab; 3] = [Activity, Readouts, Graph];
    const OLD_RIGHT: [DockTab; 5] = [Model, Knowledge, Orbitals, Selection, Log];
    /// The old sweep constant's order (`DockTab::ALL` before docking Task 2: View was not a tab).
    const OLD_ALL: [DockTab; 8] = [Activity, Readouts, Graph, Model, Knowledge, Orbitals, Selection, Log];

    /// Spec D-9's example, verbatim: the D1 default.
    const D9: &str = "H(0.24,V(0.55,[Model*,Knowledge],[Selection*,Readouts]),H(0.70,3D,V(0.62,[View*,Orbitals],[Log*,Activity,Graph])))";

    fn leaf(tabs: &[DockTab], active: usize) -> Node {
        Node::Leaf { tabs: tabs.to_vec(), active }
    }

    /// Every panel exactly once.
    fn assert_all_once(n: &Node) {
        let p = panels(n);
        for t in DockTab::PANELS {
            assert_eq!(p.iter().filter(|&&x| x == t).count(), 1, "{t:?} once in {}", encode(n));
        }
        assert_eq!(p.len(), DockTab::PANELS.len(), "{}", encode(n));
    }

    #[test]
    fn default_encodes_to_the_spec_string_and_round_trips() {
        let d = default_layout();
        assert_eq!(encode(&d), D9);
        assert_eq!(parse(D9), Some(d.clone()));
        assert_eq!(parse(&encode(&d)), Some(d.clone()));
        assert_all_once(&d);
        assert_eq!(panels(&d), vec![Model, Knowledge, Selection, Readouts, View, Orbitals, Log, Activity, Graph], "reading order: depth first");
    }

    #[test]
    fn garbage_does_not_parse() {
        for s in ["", "   ", "hello", "H(", "H(0.5,3D)", "H(x,3D,[Log])", "[Log", "3D)", "3D 3D", "V(0.5,3D,[Log]", "Q(0.5,3D,[Log])"] {
            assert_eq!(parse(s), None, "{s:?}");
        }
    }

    #[test]
    fn unknown_names_are_dropped() {
        assert_eq!(parse("H(0.3,[Model,Bogus*,Log],3D)"), Some(Node::Split { horizontal: true, frac: 0.3, a: Box::new(leaf(&[Model, Log], 0)), b: Box::new(Node::View3d) }));
    }

    #[test]
    fn names_are_case_insensitive_and_titles_are_read() {
        let want = Node::Split { horizontal: false, frac: 0.5, a: Box::new(Node::View3d), b: Box::new(leaf(&[Model, Knowledge, Log], 1)) };
        assert_eq!(parse("v(0.5,3d,[model,KNOWLEDGE*,lOg])"), Some(want.clone()));
        assert_eq!(parse("V(0.5,3D,[Model · estimated solver,Knowledge · what the model knows*,Log])"), Some(want));
    }

    #[test]
    fn a_duplicate_keeps_its_first_place() {
        let got = parse("H(0.3,[Log,Model*],V(0.5,3D,[Model,Graph,Log*]))").unwrap();
        assert_eq!(
            got,
            Node::Split {
                horizontal: true,
                frac: 0.3,
                a: Box::new(leaf(&[Log, Model], 1)),
                b: Box::new(Node::Split { horizontal: false, frac: 0.5, a: Box::new(Node::View3d), b: Box::new(leaf(&[Graph], 0)) }),
            }
        );
        // a second 3D view is a duplicate too
        assert_eq!(parse("H(0.3,3D,3D)"), Some(Node::View3d));
    }

    #[test]
    fn an_empty_leaf_is_removed_and_its_split_collapses() {
        assert_eq!(parse("H(0.3,[Bogus],3D)"), Some(Node::View3d));
        assert_eq!(parse("H(0.3,[],V(0.5,3D,[Log]))"), Some(Node::Split { horizontal: false, frac: 0.5, a: Box::new(Node::View3d), b: Box::new(leaf(&[Log], 0)) }));
        assert_eq!(parse("V(0.5,3D,H(0.5,[],[Nope]))"), Some(Node::View3d), "a split of two empty leaves goes too");
        assert_eq!(clean(leaf(&[], 0)), None);
    }

    #[test]
    fn frac_clamps() {
        let Some(Node::Split { frac, .. }) = parse("H(2.0,3D,[Log])") else { panic!() };
        assert_eq!(frac, 0.95);
        let Some(Node::Split { frac, .. }) = parse("H(-1,3D,[Log])") else { panic!() };
        assert_eq!(frac, 0.05);
        let Some(Node::Split { frac, .. }) = clean(Node::Split { horizontal: true, frac: f32::NAN, a: Box::new(Node::View3d), b: Box::new(leaf(&[Log], 0)) }) else { panic!() };
        assert_eq!(frac, 0.5, "NaN takes the middle");
    }

    #[test]
    fn a_tree_without_the_3d_view_does_not_parse() {
        assert_eq!(parse("[Model*,Log]"), None);
        assert_eq!(parse("H(0.5,[Model],[Log])"), None);
        // clean keeps a tree without 3D (a detached window's, spec D-12)
        assert_eq!(clean(leaf(&[Model], 0)), Some(leaf(&[Model], 0)));
    }

    #[test]
    fn whitespace_is_tolerated() {
        let spaced = " H ( 0.24 , V( 0.55,[ Model * , Knowledge ],[Selection*, Readouts]) ,\n H(0.70, 3D ,V(0.62,[View*,Orbitals],[ Log* ,Activity, Graph] ) ) ) ";
        assert_eq!(parse(spaced), Some(default_layout()));
    }

    #[test]
    fn the_active_tab_follows_its_star() {
        let Some(Node::Split { b, .. }) = parse("V(0.5,3D,[Model,Graph,Log*])") else { panic!() };
        assert_eq!(*b, leaf(&[Model, Graph, Log], 2));
        let Some(Node::Split { b, .. }) = parse("V(0.5,3D,[Model,Graph,Log])") else { panic!() };
        assert_eq!(*b, leaf(&[Model, Graph, Log], 0), "no star: the first");
        let Some(Node::Split { b, .. }) = parse("V(0.5,3D,[Model,Bogus*,Graph])") else { panic!() };
        assert_eq!(*b, leaf(&[Model, Graph], 0), "the starred name was dropped: the first");
        // clean keeps the active tab when tabs before it go
        assert_eq!(clean(Node::Split { horizontal: true, frac: 0.5, a: Box::new(leaf(&[Log], 0)), b: Box::new(Node::Split { horizontal: true, frac: 0.5, a: Box::new(Node::View3d), b: Box::new(leaf(&[Log, Model, Graph], 2)) }) }),
            Some(Node::Split { horizontal: true, frac: 0.5, a: Box::new(leaf(&[Log], 0)), b: Box::new(Node::Split { horizontal: true, frac: 0.5, a: Box::new(Node::View3d), b: Box::new(leaf(&[Model, Graph], 1)) }) }));
        // an out-of-range active is the first
        assert_eq!(clean(leaf(&[Model, Log], 7)), Some(leaf(&[Model, Log], 0)));
    }

    #[test]
    fn deep_nesting_is_refused_not_a_stack_overflow() {
        let s = format!("{}3D{}", "H(0.5,[Log],".repeat(10_000), ")".repeat(10_000));
        assert_eq!(parse(&s), None);
    }

    #[test]
    fn from_the_old_docks_default() {
        let n = from_sides(&OLD_LEFT, &OLD_RIGHT);
        assert_all_once(&n);
        assert_eq!(encode(&n), "H(0.24,[Activity*,Readouts,Graph],H(0.70,3D,V(0.62,[Model*,View,Knowledge,Orbitals,Selection],[Log*])))");
        assert_eq!(parse(&encode(&n)), Some(n));
    }

    #[test]
    fn from_sides_places_every_panel_once() {
        // the profile test's moved Log: it stays left, so the right column is one leaf
        let n = from_sides(&[Activity, Readouts, Graph, Log], &[Model, Orbitals, Selection]);
        assert_all_once(&n);
        assert_eq!(encode(&n), "H(0.24,[Activity*,Readouts,Graph,Log],H(0.70,3D,[Model*,View,Orbitals,Selection,Knowledge]))", "Knowledge was never listed: it joins the last right leaf");
        // nothing on the left: no left column
        let n = from_sides(&[], &OLD_ALL);
        assert_all_once(&n);
        assert_eq!(encode(&n), "H(0.70,3D,V(0.62,[Activity*,View,Readouts,Graph,Model,Knowledge,Orbitals,Selection],[Log*]))");
        // nothing on the right: View alone leads the right column, the missing panels join it
        let n = from_sides(&[Model], &[]);
        assert_all_once(&n);
        assert_eq!(encode(&n), "H(0.24,[Model*],H(0.70,3D,[View*,Knowledge,Selection,Readouts,Orbitals,Log,Activity,Graph]))", "missing panels in PANELS order");
        // only Log on the right: View leads the first leaf, the missing panels join [Log]
        let n = from_sides(&[Activity], &[Log]);
        assert_all_once(&n);
        assert_eq!(encode(&n), "H(0.24,[Activity*],H(0.70,3D,V(0.62,[View*],[Log*,Model,Knowledge,Selection,Readouts,Orbitals,Graph])))");
        // duplicates across and within the lists keep their first place
        let n = from_sides(&[Log, Log, Graph], &[Graph, Model, Log]);
        assert_all_once(&n);
        assert!(encode(&n).starts_with("H(0.24,[Log*,Graph],H(0.70,3D,[Model*,View,"), "{}", encode(&n));
        // a list that already holds View keeps it there
        let n = from_sides(&[View], &[]);
        assert_all_once(&n);
        assert!(encode(&n).starts_with("H(0.24,[View*],H(0.70,3D,["), "{}", encode(&n));
    }

    #[test]
    fn a_panels_name_is_the_profiles_word() {
        for t in DockTab::PANELS {
            assert_eq!(t.name(), format!("{t:?}"), "the profile's left_tabs / right_tabs word");
        }
    }

    #[test]
    fn from_two_empty_lists_is_the_default() {
        assert_eq!(from_sides(&[], &[]), default_layout());
    }

    #[test]
    fn hidden_panels_round_trip() {
        let h = vec![Hidden { tab: Graph, edge: Edge::Left, size: 340.0 }, Hidden { tab: Log, edge: Edge::Right, size: 300.0 }, Hidden { tab: View, edge: Edge::Bottom, size: 260.4 }];
        assert_eq!(encode_hidden(&h), "Graph:left:340, Log:right:300, View:bottom:260");
        assert_eq!(parse_hidden("Graph:left:340, Log:right:300"), h[..2].to_vec(), "the spec's example");
        assert_eq!(encode_hidden(&parse_hidden(&encode_hidden(&h))), encode_hidden(&h));
        assert_eq!(encode_hidden(&[]), "");
        assert_eq!(parse_hidden(""), vec![]);
    }

    #[test]
    fn hidden_names_and_edges_are_case_insensitive_sizes_clamped_unknown_dropped() {
        assert_eq!(
            parse_hidden(" graph : LEFT : 340 ,LOG:Bottom:5000, model:right:12"),
            vec![Hidden { tab: Graph, edge: Edge::Left, size: 340.0 }, Hidden { tab: Log, edge: Edge::Bottom, size: HIDDEN_MAX }, Hidden { tab: Model, edge: Edge::Right, size: HIDDEN_MIN }]
        );
        // an unknown name, an unknown edge, an unreadable size, a missing or extra field: dropped
        assert_eq!(parse_hidden("Bogus:left:300, Graph:top:300, Log:left:wide, Model:left, View:left:300:9, Orbitals:left:NaN, Readouts:right:inf"), vec![Hidden { tab: Readouts, edge: Edge::Right, size: HIDDEN_MAX }]);
        // a title reads as its panel; a duplicate keeps its first entry
        assert_eq!(parse_hidden("Model · estimated solver:left:300, Model:right:400"), vec![Hidden { tab: Model, edge: Edge::Left, size: 300.0 }]);
        assert_eq!(parse_hidden("garbage, ,:::,"), vec![]);
    }
}
