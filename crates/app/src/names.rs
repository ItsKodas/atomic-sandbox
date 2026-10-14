//! Molecule labels: displayed formulas, structural names read off the bond graph, the trivial-name
//! table (spec §4), and the alternative name styles the viewer offers (user request 2026-10-15:
//! "only the scientific wording, for example water is dihydrogen monoxide" — [`NameStyle`]).
//!
//! **Display data only (rule 1).** Element names, symbols, the formula order and the trivial-name table
//! map a *result* — the bond graph the density produced — to a label and feed nothing back, the same
//! status as `render::elements`' colours and symbols. They live in `app`, never in `sim` or `qm`, and the
//! simulation never reads them.
//!
//! References: E. A. Hill, J. Am. Chem. Soc. 22, 478 (1900) (Hill order, used for carbon compounds);
//! IUPAC, *Nomenclature of Inorganic Chemistry* (Red Book, 2005), IR-4.4.2.1 and Table VI (the
//! electronegativity sequence used for every other formula and for binary names; IR-5 stoichiometric
//! names; IR-6 parent hydrides).
use render::elements::symbol;
use std::collections::BTreeMap;

/// Subscript digits for an atom count; `""` for 1 (the count is omitted).
pub fn subscript(n: usize) -> String {
    if n == 1 { String::new() } else { subscript_digits(n) }
}

/// `n`'s decimal digits mapped through a ten-glyph table (sub- or superscript digits).
fn map_digits(n: impl ToString, digits: &[char; 10]) -> String {
    n.to_string().chars().map(|c| digits[c.to_digit(10).unwrap_or(0) as usize]).collect()
}

/// Subscript digits for any `n` (1 included: "C₁ hydrocarbon").
fn subscript_digits(n: usize) -> String {
    map_digits(n, &['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'])
}

/// Superscript digits (the isotope's mass number: ¹⁶O).
pub fn superscript_digits(n: u32) -> String {
    map_digits(n, &['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'])
}

/// The charge as a superscript: `""` for 0, `"⁺"`, `"⁻"`, `"²⁺"`, `"³⁻"` (magnitude 1 omitted).
pub fn superscript_charge(q: i32) -> String {
    if q == 0 {
        return String::new();
    }
    let m = q.unsigned_abs();
    let digits = if m == 1 { String::new() } else { superscript_digits(m) };
    format!("{digits}{}", if q > 0 { '⁺' } else { '⁻' })
}

/// Sorted (Z, count) pairs — the composition key every lookup uses (spec §4.2, MUST-6). Counts saturate
/// at 255 (far beyond any scene the sandbox can solve).
pub fn composition(z: &[u8]) -> Vec<(u8, u8)> {
    let mut counts: BTreeMap<u8, u8> = BTreeMap::new();
    for &zi in z {
        let c = counts.entry(zi).or_insert(0);
        *c = c.saturating_add(1);
    }
    counts.into_iter().collect()
}

/// Concatenates `symbol count` in the given element order, then the charge superscript if `Some`.
fn write_formula(comp: &[(u8, u8)], charge: Option<i32>) -> String {
    let mut s: String = comp.iter().map(|&(z, n)| format!("{}{}", symbol(z), subscript(n as usize))).collect();
    if let Some(q) = charge {
        s.push_str(&superscript_charge(q));
    }
    s
}

/// Hill order (C, H, then alphabetical; all alphabetical without C) — kept for the record, not displayed.
pub fn hill_formula(z: &[u8], charge: Option<i32>) -> String {
    let mut comp = composition(z);
    let has_carbon = comp.iter().any(|&(z, _)| z == 6);
    comp.sort_by_key(|&(z, _)| match (has_carbon, z) {
        (true, 6) => (0, ""),
        (true, 1) => (1, ""),
        _ => (2, symbol(z)),
    });
    write_formula(&comp, charge)
}

/// The IUPAC Red Book 2005 Table VI sequence for Z ≤ 10, from the ELECTROPOSITIVE end (display data):
/// Ne, He, Li, Be, B, C, N, H, O, F — position 0 for Ne. Beyond the table (Z = 0 or Z > 10) the position
/// is 10 + Z, after every listed element, so the order stays total.
pub fn iupac_order(z: u8) -> usize {
    const SEQUENCE: [u8; 10] = [10, 2, 3, 4, 5, 6, 7, 1, 8, 9];
    SEQUENCE.iter().position(|&s| s == z).unwrap_or(10 + z as usize)
}

/// The DISPLAYED formula (spec §4.1, P9): Hill order when carbon is present, `iupac_order` otherwise;
/// subscripts; the charge superscript when `charge` is `Some` (and nonzero).
pub fn formula(z: &[u8], charge: Option<i32>) -> String {
    if z.contains(&6) {
        return hill_formula(z, charge);
    }
    let mut comp = composition(z);
    comp.sort_by_key(|&(z, _)| iupac_order(z));
    write_formula(&comp, charge)
}

/// Element name for the labels and the atom card (display-only: `units::elements::name`); "?" where
/// that table has none.
pub fn element_name(z: u8) -> &'static str {
    units::elements::name(z).unwrap_or("?")
}

/// The "-ide" stem for binary names (spec §4.2): "hydr" → hydride, "ox" → oxide, …; "?" beyond Ne.
pub fn ide_stem(z: u8) -> &'static str {
    ["?", "hydr", "hel", "lith", "beryll", "bor", "carb", "nitr", "ox", "fluor", "neon"]
        .get(z as usize)
        .copied()
        .unwrap_or("?")
}

/// Multiplying prefix for a count (spec §4.2); "" for 1 (mono is omitted by the structural rule).
fn prefix(n: usize) -> &'static str {
    ["", "", "di", "tri", "tetra", "penta", "hexa", "hepta", "octa", "nona", "deca"].get(n).copied().unwrap_or("poly")
}

/// The graph signature a trivial-table entry may require besides its composition and charge (spec §4.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Signature {
    /// Composition and charge are enough.
    Any,
    /// An O–O bond with both oxygens bonded to something else as well.
    Peroxide,
    /// At least one C–O bond.
    CarbonOxygenBond,
    /// A multi-centre bond in the component (the detector's ring, `sim::bonds::Bond::multi_centre`):
    /// the trihydrogen cation is the three-centre ring, so three H⁺-charged hydrogens bonded as an
    /// ordinary triangle or chain are not given its name (M6 Task 4 ruling).
    MultiCentre,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NameKind {
    /// A hit in [`TRIVIAL`].
    Trivial,
    /// Built by a structural rule (§4.2 items 2–5); the UI marks it "(systematic)".
    Structural,
    /// No name: the formula is the whole label.
    None,
}

/// A component's label (spec §4.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Name {
    pub formula: String,
    pub text: Option<String>,
    pub kind: NameKind,
    /// The component contains a multi-centre bond (§3.5): "(ring)" goes on the tooltip, never on the
    /// formula. Set by [`label_components`]; [`classify`] (which sees plain bonds) leaves it false.
    pub ring: bool,
    /// The IR-5 stoichiometric name ([`stoichiometric_name`]) and the IR-6 parent-hydride name
    /// ([`hydride_name`]) of the same composition and charge, for the other [`NameStyle`]s.
    pub stoichiometric: Option<String>,
    pub hydride: Option<String>,
}

/// How a molecule is named in the view (a viewer preference; the formula is always shown).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NameStyle {
    /// Trivial names where the table has one, else the structural rules ("water", "ammonia",
    /// "helium dihydride (systematic)") — the default.
    #[default]
    Common,
    /// IUPAC Red Book 2005 IR-5 stoichiometric names ("dihydrogen monoxide", "nitrogen trihydride").
    Stoichiometric,
    /// IR-6 parent-hydride names where one exists ("oxidane", "azane", "methane"), else the
    /// stoichiometric name.
    Hydride,
    /// The formula alone.
    FormulaOnly,
}

impl NameStyle {
    pub const ALL: [NameStyle; 4] = [NameStyle::Common, NameStyle::Stoichiometric, NameStyle::Hydride, NameStyle::FormulaOnly];

    pub fn title(self) -> &'static str {
        match self {
            NameStyle::Common => "common",
            NameStyle::Stoichiometric => "systematic (stoichiometric)",
            NameStyle::Hydride => "IUPAC hydride",
            NameStyle::FormulaOnly => "formula only",
        }
    }
}

impl Name {
    /// The label next to the molecule (spec §4.4): "H₂O — water", "HeH₂ — helium dihydride (systematic)",
    /// or the formula alone.
    pub fn label(&self) -> String {
        match (&self.text, &self.kind) {
            (Some(t), NameKind::Structural) => format!("{} — {t} (systematic)", self.formula),
            (Some(t), _) => format!("{} — {t}", self.formula),
            (None, _) => self.formula.clone(),
        }
    }

    /// The tooltip's title line: the label plus " (ring)" for a component with a multi-centre bond.
    pub fn tooltip(&self) -> String {
        if self.ring { format!("{} (ring)", self.label()) } else { self.label() }
    }

    /// The label in a [`NameStyle`]: "H₂O — water", "H₂O — dihydrogen monoxide", "H₂O — oxidane", "H₂O".
    /// A style with no name for the composition (a ternary compound's stoichiometric name, a
    /// heteronuclear ion's) shows the formula alone.
    pub fn label_in(&self, style: NameStyle) -> String {
        let with = |t: &Option<String>| t.as_ref().map_or_else(|| self.formula.clone(), |t| format!("{} — {t}", self.formula));
        match style {
            NameStyle::Common => self.label(),
            NameStyle::Stoichiometric => with(&self.stoichiometric),
            NameStyle::Hydride => with(&self.hydride.clone().or_else(|| self.stoichiometric.clone())),
            NameStyle::FormulaOnly => self.formula.clone(),
        }
    }

    /// [`Name::tooltip`] in a style: the label in it, " (ring)", and the common name when it differs.
    pub fn tooltip_in(&self, style: NameStyle) -> String {
        if style == NameStyle::Common {
            return self.tooltip();
        }
        let mut t = self.label_in(style);
        if self.ring {
            t += " (ring)";
        }
        if self.text.is_some() && self.label() != self.label_in(style) {
            t += &format!("\ncommon name: {}", self.label());
        }
        t
    }
}

/// An IUPAC charge number for an ion's name (Red Book IR-5.4): "(1+)", "(2−)"; "" for 0.
fn charge_number(q: i32) -> String {
    match q {
        0 => String::new(),
        q => format!("({}{})", q.unsigned_abs(), if q > 0 { '+' } else { '−' }),
    }
}

/// The IR-5 stoichiometric name of a composition (spec addendum, user request 2026-10-15), built from
/// the counts only — no structure:
/// - one element: the multiplicative prefix and the element name ("dihydrogen", "dioxygen",
///   "monohydrogen" for one neutral atom, IR-3.4.1); a cation adds its charge number
///   ("trihydrogen(1+)"; a lone atom's without "mono", IR-5.3.2.2: "helium(1+)"), an anion takes the
///   "-ide" stem ("dioxide(2−)", "hydride(1−)");
/// - two elements, neutral: the electropositive constituent first (the Red Book Table VI order the
///   formulas use), [`binary_name`]: "dihydrogen monoxide", "carbon dioxide", "nitrogen trihydride",
///   "lithium hydride".
/// - anything else (a heteronuclear ion such as HeH⁺, whose IUPAC name is additive —
///   "hydridohelium(1+)" — or a compound of three or more elements): `None`, the formula alone.
pub fn stoichiometric_name(comp: &[(u8, u8)], charge: i32) -> Option<String> {
    match comp {
        [(z, n)] => {
            let p = match (*n, charge) {
                (1, 0) => "mono",
                _ => prefix(*n as usize),
            };
            Some(if charge < 0 {
                format!("{p}{}ide{}", ide_stem(*z), charge_number(charge))
            } else {
                format!("{p}{}{}", element_name(*z), charge_number(charge))
            })
        }
        [a, b] if charge == 0 => {
            let (&(za, m), &(zb, n)) = if iupac_order(a.0) < iupac_order(b.0) { (a, b) } else { (b, a) };
            Some(binary_name(za, m, zb, n))
        }
        _ => None,
    }
}

/// The binary name A_m B_n with A the electropositive constituent (IR-5.2), shared by the
/// stoichiometric style and the Common style's structural fallback (§4.2 rule 5), so a compound
/// reads the same in both: the multiplicative prefixes on both constituents, "mono" never on the
/// first; on the second it is written for an oxide or when the first has a prefix ("carbon
/// monoxide", "dinitrogen monoxide", "dihydrogen monoxide", but "hydrogen fluoride", "lithium
/// hydride") — the Red Book omits "mono" where nothing is ambiguous, and this fixed rule is our
/// reading of its examples. Elision: only "monoxide" (IR-5.2 keeps every other prefix vowel:
/// "tetraoxide", not "tetroxide").
pub fn binary_name(za: u8, m: u8, zb: u8, n: u8) -> String {
    let second = match n {
        1 if m > 1 || zb == 8 => "mono",
        _ => prefix(n as usize),
    };
    let tail = format!("{second}{}ide", ide_stem(zb)).replace("monoox", "monox");
    format!("{}{} {tail}", prefix(m as usize), element_name(za))
}

/// IR-6 parent hydrides of the compositions in range (Red Book 2005 Table IR-6.1; their onium and
/// "-ide" ions, IR-6.4): mononuclear BH₃ borane, CH₄ methane, NH₃ azane, H₂O oxidane, HF fluorane;
/// the chains H₂O₂ dioxidane, N₂H₄ diazane, C₂H₆ ethane; NH₄⁺ azanium, H₃O⁺ oxidanium, HO⁻ oxidanide.
/// A chain name asserts its backbone, so the graph must have one — the checks the Common style uses:
/// dioxidane the peroxide signature (an O–O bond, both O further bonded), ethane the carbons
/// connected through C–C bonds, diazane an N–N bond. `None` for anything else, or for a chain
/// composition without its backbone (the stoichiometric name stands in).
#[cfg(test)] // the app reaches it through classify_graph; the graph-level entry for the tests
pub fn hydride_name(z: &[u8], bonds: &[(usize, usize)], charge: i32) -> Option<String> {
    hydride_of(z, &adjacency(z.len(), bonds), &composition(z), charge)
}

fn hydride_of(z: &[u8], adj: &[Vec<usize>], comp: &[(u8, u8)], charge: i32) -> Option<String> {
    let name = match (comp, charge) {
        ([(1, 3), (5, 1)], 0) => "borane",
        ([(1, 4), (6, 1)], 0) => "methane",
        ([(1, 3), (7, 1)], 0) => "azane",
        ([(1, 2), (8, 1)], 0) => "oxidane",
        ([(1, 1), (9, 1)], 0) => "fluorane",
        ([(1, 2), (8, 2)], 0) if has_peroxide(z, adj) => "dioxidane",
        ([(1, 4), (7, 2)], 0) if connected_among(adj, |i| z[i] == 7) => "diazane",
        ([(1, 6), (6, 2)], 0) if carbons_connected(z, adj) => "ethane",
        ([(1, 4), (7, 1)], 1) => "azanium",
        ([(1, 3), (8, 1)], 1) => "oxidanium",
        ([(1, 1), (8, 1)], -1) => "oxidanide",
        _ => return None,
    };
    Some(name.to_string())
}

/// The trivial-name table keyed by COMPOSITION (sorted (Z, count)) + charge + graph signature (spec §4.3)
/// — never by a formula string, so a change of display order cannot break a lookup (MUST-6).
#[rustfmt::skip]
pub const TRIVIAL: &[(&[(u8, u8)], i32, Signature, &str)] = &[
    (&[(1, 2)], 0, Signature::Any, "dihydrogen"),
    (&[(8, 2)], 0, Signature::Any, "dioxygen"),
    (&[(7, 2)], 0, Signature::Any, "dinitrogen"),
    (&[(9, 2)], 0, Signature::Any, "difluorine"),
    (&[(3, 2)], 0, Signature::Any, "dilithium"),
    (&[(1, 2), (8, 1)], 0, Signature::Any, "water"),
    (&[(1, 2), (8, 2)], 0, Signature::Peroxide, "hydrogen peroxide"),
    (&[(1, 3), (7, 1)], 0, Signature::Any, "ammonia"),
    (&[(1, 4), (7, 1)], 1, Signature::Any, "ammonium"),
    (&[(1, 4), (6, 1)], 0, Signature::Any, "methane"),
    (&[(1, 6), (6, 2)], 0, Signature::Any, "ethane"),
    (&[(1, 4), (6, 2)], 0, Signature::Any, "ethene"),
    (&[(1, 2), (6, 2)], 0, Signature::Any, "ethyne"),
    (&[(1, 1), (9, 1)], 0, Signature::Any, "hydrogen fluoride"),
    (&[(1, 1), (3, 1)], 0, Signature::Any, "lithium hydride"),
    (&[(3, 1), (9, 1)], 0, Signature::Any, "lithium fluoride"),
    (&[(1, 2), (4, 1)], 0, Signature::Any, "beryllium hydride"),
    (&[(1, 3), (5, 1)], 0, Signature::Any, "borane"),
    (&[(1, 3)], 1, Signature::MultiCentre, "trihydrogen cation"),
    (&[(1, 1), (2, 1)], 1, Signature::Any, "helium hydride cation"),
    (&[(1, 1), (8, 1)], -1, Signature::Any, "hydroxide"),
    (&[(1, 1), (8, 1)], 0, Signature::Any, "hydroxyl"),
    (&[(1, 3), (8, 1)], 1, Signature::Any, "hydronium"),
    (&[(6, 1), (8, 1)], 0, Signature::Any, "carbon monoxide"),
    (&[(6, 1), (8, 2)], 0, Signature::Any, "carbon dioxide"),
    (&[(7, 1), (8, 1)], 0, Signature::Any, "nitric oxide"),
    (&[(1, 1), (6, 1), (7, 1)], 0, Signature::Any, "hydrogen cyanide"),
    (&[(1, 4), (6, 1), (8, 1)], 0, Signature::CarbonOxygenBond, "methanol"),
];

/// Adjacency lists for `n` nuclei from local-index bonds (out-of-range, self and repeated bonds ignored).
fn adjacency(n: usize, bonds: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); n];
    for &(i, j) in bonds {
        if i < n && j < n && i != j && !adj[i].contains(&j) {
            adj[i].push(j);
            adj[j].push(i);
        }
    }
    adj
}

/// Whether the nuclei for which `keep` holds are connected using only edges between such nuclei
/// (true for zero or one of them).
fn connected_among(adj: &[Vec<usize>], keep: impl Fn(usize) -> bool) -> bool {
    let members: Vec<usize> = (0..adj.len()).filter(|&i| keep(i)).collect();
    let Some(&start) = members.first() else { return true };
    let mut seen = vec![false; adj.len()];
    let mut stack = vec![start];
    seen[start] = true;
    let mut reached = 1;
    while let Some(i) = stack.pop() {
        for &j in &adj[i] {
            if !seen[j] && keep(j) {
                seen[j] = true;
                reached += 1;
                stack.push(j);
            }
        }
    }
    reached == members.len()
}

/// An O–O bond with both oxygens further bonded (spec §4.2 item 3).
fn has_peroxide(z: &[u8], adj: &[Vec<usize>]) -> bool {
    (0..z.len()).any(|i| z[i] == 8 && adj[i].iter().any(|&j| z[j] == 8 && adj[i].len() >= 2 && adj[j].len() >= 2))
}

/// Whether a table entry's graph condition holds. Besides the entry's `Signature`, the hydrocarbon entries
/// (methane … ethyne) carry rule 4's own condition — the carbons connected through C–C bonds — so a
/// C₂H₆ bridged only through hydrogen is not "ethane" (it falls through to the structural rules).
fn signature_holds(sig: Signature, z: &[u8], adj: &[Vec<usize>], comp: &[(u8, u8)], ring: bool) -> bool {
    if is_hydrocarbon(comp) && !carbons_connected(z, adj) {
        return false;
    }
    match sig {
        Signature::Any => true,
        Signature::MultiCentre => ring,
        Signature::Peroxide => has_peroxide(z, adj),
        Signature::CarbonOxygenBond => (0..z.len()).any(|i| z[i] == 6 && adj[i].iter().any(|&j| z[j] == 8)),
    }
}

/// Only C and H, at least one C.
fn is_hydrocarbon(comp: &[(u8, u8)]) -> bool {
    comp.iter().all(|&(zz, _)| zz == 1 || zz == 6) && comp.iter().any(|&(zz, _)| zz == 6)
}

/// The carbons are mutually connected through C–C bonds (spec §4.2 item 4).
fn carbons_connected(z: &[u8], adj: &[Vec<usize>]) -> bool {
    connected_among(adj, |i| z[i] == 6)
}

/// " cation", " dication", " anion", … appended to a structural name; "" for neutral.
fn charge_suffix(q: i32) -> String {
    match q {
        0 => String::new(),
        q => format!(" {}{}", prefix(q.unsigned_abs() as usize), if q > 0 { "cation" } else { "anion" }),
    }
}

/// The structural rules of spec §4.2 items 2–5 (the table already missed). `None` when no rule applies.
fn structural_name(z: &[u8], adj: &[Vec<usize>], comp: &[(u8, u8)]) -> Option<String> {
    // 2. homonuclear diatomic X₂
    if let [(x, 2)] = comp {
        return Some(format!("di{}", element_name(*x)));
    }
    // 3. peroxide: name the rest when it is a single element other than oxygen ("dilithium peroxide");
    //    a hydrogen rest is "hydrogen peroxide" only as H₂ (any other count: the class name "peroxide");
    //    an all-oxygen rest (an O₃ ring, an O₄ chain) is not a peroxide of anything — the rule does not apply
    if has_peroxide(z, adj) {
        let rest: Vec<(u8, u8)> = comp
            .iter()
            .map(|&(zz, n)| if zz == 8 { (zz, n - 2) } else { (zz, n) })
            .filter(|&(_, n)| n > 0)
            .collect();
        match rest.as_slice() {
            [(8, _)] => {}
            [(1, 2)] => return Some("hydrogen peroxide".to_string()),
            [(1, _)] => return Some("peroxide".to_string()),
            [(a, m)] => return Some(format!("{}{} peroxide", prefix(*m as usize), element_name(*a))),
            _ => return Some("peroxide".to_string()),
        }
    }
    // 4. simple hydrocarbon: only C and H, the carbons connected through C–C bonds (otherwise rule 5)
    if is_hydrocarbon(comp) && carbons_connected(z, adj) {
        let count = |e: u8| comp.iter().find(|&&(zz, _)| zz == e).map_or(0, |&(_, n)| n as usize);
        let (n_c, n_h) = (count(6), count(1));
        let named = match (n_c, n_h) {
            (1, 4) => Some("methane"),
            (2, 6) => Some("ethane"),
            (2, 4) => Some("ethene"),
            (2, 2) => Some("ethyne"),
            _ => None,
        };
        return Some(named.map_or_else(|| format!("C{} hydrocarbon", subscript_digits(n_c)), str::to_string));
    }
    // 5. binary A_m B_n: A earlier in the Table VI sequence; the shared IR-5 builder, except that no
    //    prefixes follow a hydrogen first constituent ("hydrogen oxide", spec §4.2)
    if let [p, q] = comp {
        let (&(a, m), &(b, n)) = if iupac_order(p.0) < iupac_order(q.0) { (p, q) } else { (q, p) };
        if a == 1 {
            return Some(format!("{} {}ide", element_name(a), ide_stem(b)));
        }
        return Some(binary_name(a, m, b, n));
    }
    None
}

/// One component: its nuclei's Z (in scene order), the bonds among them (local indices), the charge to
/// show (spec §4.1 rule applied by the caller; `None` = unassigned, looked up as neutral). Precedence
/// (spec §4.2): trivial table on composition + charge + signature, homonuclear diatomic, peroxide,
/// simple hydrocarbon, binary compound, else the formula alone. Atoms the given bonds do not connect
/// get no name (a component is connected by construction; this guards other callers). Plain bonds
/// carry no multi-centre flag, so a [`Signature::MultiCentre`] entry never matches here: use
/// [`classify_graph`] with the detector's flag (the app does: `label_components`; this plain form is
/// the spec's §4.2 signature, kept for the tests).
#[cfg(test)]
pub fn classify(z: &[u8], bonds: &[(usize, usize)], charge: Option<i32>) -> Name {
    classify_graph(z, bonds, false, charge)
}

/// [`classify`] with `ring`: some bond of the component is multi-centre (the detector's ring).
/// The returned `Name::ring` is that flag.
pub fn classify_graph(z: &[u8], bonds: &[(usize, usize)], ring: bool, charge: Option<i32>) -> Name {
    let formula = formula(z, charge);
    let comp = composition(z);
    let q = charge.unwrap_or(0);
    let adj = adjacency(z.len(), bonds);
    let (stoichiometric, hydride) = (stoichiometric_name(&comp, q), hydride_of(z, &adj, &comp, q));
    let named = |formula, text, kind| Name { formula, text, kind, ring, stoichiometric: stoichiometric.clone(), hydride: hydride.clone() };
    let unnamed = |formula| named(formula, None, NameKind::None);
    if z.len() < 2 || !connected_among(&adj, |_| true) {
        // a lone atom keeps its stoichiometric name ("monohydrogen"); an unconnected set has none
        let lone = z.len() == 1;
        return Name { stoichiometric: stoichiometric.clone().filter(|_| lone), hydride: None, ..unnamed(formula) };
    }
    if let Some(&(_, _, _, name)) =
        TRIVIAL.iter().find(|&&(key, c, sig, _)| key == comp.as_slice() && c == q && signature_holds(sig, z, &adj, &comp, ring))
    {
        return named(formula, Some(name.to_string()), NameKind::Trivial);
    }
    match structural_name(z, &adj, &comp) {
        Some(text) => named(formula, Some(text + &charge_suffix(q)), NameKind::Structural),
        None => unnamed(formula),
    }
}

/// One bond of a [`MoleculeGraph`] (global nucleus indices).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphBond {
    pub i: usize,
    pub j: usize,
    pub multi_centre: bool,
}

/// Plain-data view of the bond graph that [`label_components`] reads: the bonds and the connected
/// components (sorted index lists, isolated nuclei included). The worker builds it from
/// `sim::bonds::BondGraph` (`worker::molecule_graph`), keeping `app::names` free of the detector's
/// types.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MoleculeGraph {
    pub bonds: Vec<GraphBond>,
    pub components: Vec<Vec<usize>>,
}

/// Labels for every component of a graph: one `Name` per component, in `graph.components` order.
/// Applies the single-component charge rule (spec §4.1, P1): the net charge is shown on the formula only
/// when the scene has exactly one component; with several and a nonzero net charge every formula is
/// neutral-looking (names looked up as neutral) and the scene-level note is returned.
pub fn label_components(z_all: &[u8], graph: &MoleculeGraph, net_charge: i32) -> (Vec<Name>, Option<String>) {
    // assignable: one component (or none — an empty scene has no molecule to leave the charge off)
    let assignable = graph.components.len() <= 1 || net_charge == 0;
    let charge = if assignable { Some(net_charge) } else { None };
    let names = graph
        .components
        .iter()
        .map(|members| {
            let local = |g: usize| members.iter().position(|&m| m == g);
            let z: Vec<u8> = members.iter().map(|&g| z_all[g]).collect();
            let mut bonds = Vec::new();
            let mut ring = false;
            for b in &graph.bonds {
                if let (Some(i), Some(j)) = (local(b.i), local(b.j)) {
                    bonds.push((i, j));
                    ring |= b.multi_centre;
                }
            }
            classify_graph(&z, &bonds, ring, charge)
        })
        .collect();
    let note = (charge.is_none()).then(|| {
        let sign = if net_charge > 0 { '+' } else { '−' };
        format!("net charge {sign}{} — not assignable to one molecule without a density partition", net_charge.unsigned_abs())
    });
    (names, note)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hill_formula_is_strict_hill_and_is_not_what_we_display() {
        assert_eq!(hill_formula(&[3, 1], None), "HLi");
        assert_eq!(hill_formula(&[2, 1], Some(1)), "HHe⁺");
        assert_eq!(hill_formula(&[1, 9], None), "FH");
        assert_eq!(hill_formula(&[6, 1, 7], None), "CHN");
    }

    #[test]
    fn displayed_formulas_follow_hill_for_carbon_and_the_iupac_sequence_otherwise() {
        assert_eq!(formula(&[1, 1, 8], Some(0)), "H₂O");
        assert_eq!(formula(&[3, 1], Some(0)), "LiH");
        assert_eq!(formula(&[1, 9], Some(0)), "HF");
        assert_eq!(formula(&[7, 1, 1, 1], Some(0)), "NH₃");
        assert_eq!(formula(&[2, 1], Some(1)), "HeH⁺");
        assert_eq!(formula(&[1, 1, 1], Some(1)), "H₃⁺");
        assert_eq!(formula(&[8, 1], Some(-1)), "HO⁻");
        assert_eq!(formula(&[8, 8], Some(-2)), "O₂²⁻");
        assert_eq!(formula(&[6, 1, 1, 1, 1], None), "CH₄");
        assert_eq!(formula(&[6, 6, 1, 1, 1, 1, 1, 1, 8], None), "C₂H₆O");
        assert_eq!(formula(&[6, 1, 7], None), "CHN");
        assert_eq!(formula(&[6, 8, 8], None), "CO₂");
        assert_eq!(iupac_order(10), 0);
        assert!(iupac_order(2) < iupac_order(1) && iupac_order(1) < iupac_order(8) && iupac_order(7) < iupac_order(1));
    }

    #[test]
    fn the_iupac_sequence_is_table_vi_from_the_electropositive_end() {
        let seq: Vec<u8> = {
            let mut z: Vec<u8> = (1..=10).collect();
            z.sort_by_key(|&z| iupac_order(z));
            z
        };
        assert_eq!(seq, vec![10, 2, 3, 4, 5, 6, 7, 1, 8, 9]); // Ne He Li Be B C N H O F
        assert_eq!(formula(&[9, 3], Some(0)), "LiF");
        assert_eq!(formula(&[8, 1, 1, 1], Some(1)), "H₃O⁺");
        assert_eq!(formula(&[9, 8, 9], Some(0)), "OF₂"); // O before F: the easy one to get backwards
        assert_eq!(formula(&[1, 6, 1, 1, 6, 1, 1, 1], Some(0)), "C₂H₆");
    }

    #[test]
    fn subscripts_and_charge_superscripts() {
        assert_eq!(subscript(1), "");
        assert_eq!(subscript(2), "₂");
        assert_eq!(subscript(12), "₁₂");
        assert_eq!(superscript_charge(0), "");
        assert_eq!(superscript_charge(1), "⁺");
        assert_eq!(superscript_charge(-1), "⁻");
        assert_eq!(superscript_charge(2), "²⁺");
        assert_eq!(superscript_charge(-3), "³⁻");
        assert_eq!(superscript_digits(16), "¹⁶");
        assert_eq!(superscript_digits(7), "⁷");
    }

    #[test]
    fn composition_keys_are_sorted_and_order_independent() {
        assert_eq!(composition(&[8, 1, 1]), vec![(1, 2), (8, 1)]);
        assert_eq!(composition(&[1, 8, 1]), composition(&[8, 1, 1]));
    }

    #[test]
    fn the_trivial_table_is_keyed_by_sorted_compositions() {
        for (comp, _, _, name) in TRIVIAL {
            let mut sorted = comp.to_vec();
            sorted.sort();
            sorted.dedup_by_key(|p| p.0);
            assert_eq!(&sorted[..], *comp, "{name}: key not sorted / not unique");
            assert!(comp.iter().all(|&(_, n)| n > 0), "{name}: zero count");
        }
        // no two entries share (composition, charge, signature)
        for (a, ea) in TRIVIAL.iter().enumerate() {
            for eb in &TRIVIAL[a + 1..] {
                assert!(!(ea.0 == eb.0 && ea.1 == eb.1 && ea.2 == eb.2), "{} / {}", ea.3, eb.3);
            }
        }
        assert_eq!(TRIVIAL.len(), 28); // spec §4.3's list
    }

    #[test]
    fn trivial_names_win_by_composition_not_by_string() {
        let n = classify(&[8, 1, 1], &[(0, 1), (0, 2)], Some(0));
        assert_eq!((n.text.as_deref(), n.kind), (Some("water"), NameKind::Trivial));
        assert_eq!(classify(&[1, 8, 1], &[(0, 1), (1, 2)], Some(0)).text.as_deref(), Some("water"));
        // the trihydrogen cation is the three-centre ring: plain bonds do not earn the name
        assert_eq!(classify(&[1, 1, 1], &[(0, 1), (1, 2), (0, 2)], Some(1)).text, None);
        let n = classify_graph(&[1, 1, 1], &[(0, 1), (1, 2), (0, 2)], true, Some(1));
        assert_eq!((n.text.as_deref(), n.kind, n.ring), (Some("trihydrogen cation"), NameKind::Trivial, true));
        let n = classify(&[2, 1], &[(0, 1)], Some(1));
        assert_eq!((n.formula.as_str(), n.text.as_deref()), ("HeH⁺", Some("helium hydride cation")));
        let n = classify(&[1, 2], &[(0, 1)], Some(1));
        assert_eq!((n.formula.as_str(), n.text.as_deref()), ("HeH⁺", Some("helium hydride cation")));
        assert_eq!(
            classify(&[6, 8, 1, 1, 1, 1], &[(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)], Some(0)).text.as_deref(),
            Some("methanol")
        );
        assert_eq!(
            classify(&[6, 8, 1, 1, 1, 1], &[(0, 2), (0, 3), (0, 4), (0, 5), (1, 5)], Some(0)).text,
            None /* same composition, no C–O bond: not methanol */
        );
    }

    #[test]
    fn peroxide_needs_both_oxygens_further_bonded() {
        assert_eq!(
            classify(&[1, 8, 8, 1], &[(0, 1), (1, 2), (2, 3)], Some(0)).text.as_deref(),
            Some("hydrogen peroxide")
        );
        assert_eq!(classify(&[8, 8], &[(0, 1)], Some(0)).text.as_deref(), Some("dioxygen"));
        // H–H–O–O: the composition of H₂O₂ but the terminal O has no further bond — not a peroxide;
        // it falls through to the binary rule (hydrogen first: no prefixes)
        let n = classify(&[1, 1, 8, 8], &[(0, 1), (1, 2), (2, 3)], Some(0));
        assert_eq!((n.text.as_deref(), n.kind), (Some("hydrogen oxide"), NameKind::Structural));
        // the structural peroxide rule: Li–O–O–Li
        let n = classify(&[3, 8, 8, 3], &[(0, 1), (1, 2), (2, 3)], Some(0));
        assert_eq!((n.formula.as_str(), n.text.as_deref(), n.kind), ("Li₂O₂", Some("dilithium peroxide"), NameKind::Structural));
        // an all-hydrogen rest other than H₂ is not "hydrogen peroxide": H–O–O(–H)–H reads as the class name
        let n = classify(&[1, 8, 8, 1, 1], &[(0, 1), (1, 2), (2, 3), (2, 4)], Some(1));
        assert_eq!((n.formula.as_str(), n.text.as_deref(), n.kind), ("H₃O₂⁺", Some("peroxide cation"), NameKind::Structural));
        // an O₃ triangle (every O bonded twice) is not a peroxide of oxygen: no rule applies
        let n = classify(&[8, 8, 8], &[(0, 1), (1, 2), (0, 2)], Some(0));
        assert_eq!((n.formula.as_str(), n.text, n.kind), ("O₃", None, NameKind::None));
    }

    #[test]
    fn hydrocarbons_by_h_count_on_the_chain() {
        assert_eq!(
            classify(&[6, 6, 1, 1, 1, 1, 1, 1], &[(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)], Some(0))
                .text
                .as_deref(),
            Some("ethane")
        );
        assert_eq!(
            classify(&[6, 6, 1, 1, 1, 1], &[(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)], Some(0)).text.as_deref(),
            Some("ethene")
        );
        assert_eq!(classify(&[6, 6, 1, 1], &[(0, 1), (0, 2), (1, 3)], Some(0)).text.as_deref(), Some("ethyne"));
        let n = classify(
            &[6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1],
            &[(0, 1), (1, 2), (0, 3), (0, 4), (0, 5), (1, 6), (1, 7), (2, 8), (2, 9), (2, 10)],
            Some(0),
        );
        assert_eq!((n.text.as_deref(), n.kind), (Some("C₃ hydrocarbon"), NameKind::Structural));
        // the structural rule itself (no table): the methane radical cation
        let n = classify(&[6, 1, 1, 1, 1], &[(0, 1), (0, 2), (0, 3), (0, 4)], Some(1));
        assert_eq!((n.formula.as_str(), n.text.as_deref(), n.kind), ("CH₄⁺", Some("methane cation"), NameKind::Structural));
        // an H count that fits no -ane/-ene/-yne
        let n = classify(&[6, 6, 1, 1, 1], &[(0, 1), (0, 2), (0, 3), (1, 4)], Some(0));
        assert_eq!((n.text.as_deref(), n.kind), (Some("C₂ hydrocarbon"), NameKind::Structural));
        // carbons not connected through C–C (C–H–C): not a simple hydrocarbon — rule 5 (binary) instead
        let n = classify(&[6, 1, 6], &[(0, 1), (1, 2)], Some(0));
        assert_eq!((n.text.as_deref(), n.kind), (Some("dicarbon monohydride"), NameKind::Structural));
        // the table's hydrocarbons need the same C–C connectivity: C₂H₆ bridged through an H is not ethane
        let n = classify(&[6, 1, 6, 1, 1, 1, 1, 1], &[(0, 1), (1, 2), (0, 3), (0, 4), (2, 5), (2, 6), (2, 7)], Some(0));
        assert_eq!((n.formula.as_str(), n.text.as_deref(), n.kind), ("C₂H₆", Some("dicarbon hexahydride"), NameKind::Structural));
    }

    #[test]
    fn binary_names_follow_the_iupac_sequence_with_prefixes() {
        assert_eq!(classify(&[3, 1], &[(0, 1)], Some(0)).text.as_deref(), Some("lithium hydride"));
        assert_eq!(classify(&[6, 8, 8], &[(0, 1), (0, 2)], Some(0)).text.as_deref(), Some("carbon dioxide"));
        assert_eq!(classify(&[7, 7, 8], &[(0, 1), (1, 2)], Some(0)).text.as_deref(), Some("dinitrogen monoxide"));
        assert_eq!(classify(&[1, 9], &[(0, 1)], Some(0)).text.as_deref(), Some("hydrogen fluoride"));
        assert_eq!(classify(&[4, 1, 1], &[(0, 1), (0, 2)], Some(0)).kind, NameKind::Trivial /* beryllium hydride */);
        let n = classify(&[2, 1, 1], &[(0, 1), (0, 2)], Some(0));
        assert_eq!(
            (n.text.as_deref(), n.kind),
            (Some("helium dihydride"), NameKind::Structural) /* no table entry: the structural rule, helium first per Table VI */
        );
        // homonuclear diatomics outside the table, and a charge on a structural name
        assert_eq!(classify(&[2, 2], &[(0, 1)], Some(0)).text.as_deref(), Some("dihelium"));
        let n = classify(&[2, 2], &[(0, 1)], Some(1));
        assert_eq!((n.formula.as_str(), n.text.as_deref()), ("He₂⁺", Some("dihelium cation")));
        assert_eq!(classify(&[8, 8], &[(0, 1)], Some(-2)).text.as_deref(), Some("dioxygen dianion"));
        assert_eq!(classify(&[6, 9, 9, 9, 9], &[(0, 1), (0, 2), (0, 3), (0, 4)], Some(0)).text.as_deref(), Some("carbon tetrafluoride"));
        assert_eq!(classify(&[9, 8, 9], &[(0, 1), (1, 2)], Some(0)).text.as_deref(), Some("oxygen difluoride"));
    }

    #[test]
    fn unbonded_and_unknown_shapes_get_a_formula_only() {
        let n = classify(&[1], &[], Some(0));
        assert_eq!((n.formula.as_str(), n.text, n.kind), ("H", None, NameKind::None));
        let n = classify(&[3, 8, 1, 1, 1], &[(0, 1), (1, 2), (1, 3), (1, 4)], Some(0));
        assert_eq!(n.kind, NameKind::None);
        // not connected by the bonds given: no name, even when the composition is in the table
        let n = classify(&[1, 1, 8], &[(0, 2)], Some(0));
        assert_eq!((n.formula.as_str(), n.text, n.kind), ("H₂O", None, NameKind::None));
    }

    fn graph(bonds: &[(usize, usize, bool)], components: &[&[usize]]) -> MoleculeGraph {
        MoleculeGraph {
            bonds: bonds.iter().map(|&(i, j, m)| GraphBond { i, j, multi_centre: m }).collect(),
            components: components.iter().map(|c| c.to_vec()).collect(),
        }
    }

    #[test]
    fn the_charge_superscript_needs_a_single_component() {
        // one component, an H₃⁺ triangle of ordinary bonds: the formula keeps its charge, but the
        // name needs the detector's three-centre ring (M6 Task 4 ruling)
        let (names, note) = label_components(&[1, 1, 1], &graph(&[(0, 1, false), (1, 2, false), (0, 2, false)], &[&[0, 1, 2]]), 1);
        assert_eq!(names.iter().map(|n| n.formula.as_str()).collect::<Vec<_>>(), vec!["H₃⁺"]);
        assert_eq!((names[0].text.as_deref(), &names[0].kind), (None, &NameKind::None));
        assert_eq!(note, None);
        // two components and a net charge: neutral-looking formulas and the scene note
        let (names, note) = label_components(&[1, 1, 1], &graph(&[(0, 1, false)], &[&[0, 1], &[2]]), 1);
        assert_eq!(names.iter().map(|n| n.formula.as_str()).collect::<Vec<_>>(), vec!["H₂", "H"]);
        assert_eq!(note.as_deref(), Some("net charge +1 — not assignable to one molecule without a density partition"));
        assert_eq!(names[0].text.as_deref(), Some("dihydrogen"));
        let (_, note) = label_components(&[1, 1, 1], &graph(&[(0, 1, false)], &[&[0, 1], &[2]]), -2);
        assert_eq!(note.as_deref(), Some("net charge −2 — not assignable to one molecule without a density partition"));
        // two components, neutral scene: no note
        let (names, note) = label_components(&[8, 1, 1, 1, 1], &graph(&[(0, 1, false), (0, 2, false), (3, 4, false)], &[&[0, 1, 2], &[3, 4]]), 0);
        assert_eq!(names.iter().map(|n| n.formula.as_str()).collect::<Vec<_>>(), vec!["H₂O", "H₂"]);
        assert_eq!(note, None);
        // a ring: " (ring)" goes on the tooltip text, never on the formula
        let (names, _) = label_components(&[1, 1, 1], &graph(&[(0, 1, true), (1, 2, true), (0, 2, true)], &[&[0, 1, 2]]), 1);
        assert_eq!(names[0].formula, "H₃⁺");
        assert!(names[0].ring);
        assert_eq!(names[0].label(), "H₃⁺ — trihydrogen cation");
        assert_eq!(names[0].tooltip(), "H₃⁺ — trihydrogen cation (ring)");
        // components index the global scene: local bond indices are remapped
        let (names, _) = label_components(&[1, 1, 8, 1], &graph(&[(2, 1, false), (2, 3, false)], &[&[0], &[1, 2, 3]]), 0);
        assert_eq!(names[1].text.as_deref(), Some("water"));
        assert!(!names[1].ring);
        // an empty graph: no molecules, no note
        assert_eq!(label_components(&[], &MoleculeGraph::default(), 1), (vec![], None));
    }

    /// The four name styles (user request 2026-10-15) on the molecules in range; the formula is in
    /// every label. Ions and homonuclear species: what each style honestly has.
    #[test]
    fn every_name_style_names_the_common_molecules() {
        use NameStyle::*;
        let tri = [(0, 1), (1, 2), (0, 2)];
        let cases: Vec<(Name, [&str; 4])> = vec![
            (classify(&[8, 1, 1], &[(0, 1), (0, 2)], Some(0)), ["H₂O — water", "H₂O — dihydrogen monoxide", "H₂O — oxidane", "H₂O"]),
            (classify(&[7, 1, 1, 1], &[(0, 1), (0, 2), (0, 3)], Some(0)), ["NH₃ — ammonia", "NH₃ — nitrogen trihydride", "NH₃ — azane", "NH₃"]),
            (classify(&[6, 1, 1, 1, 1], &[(0, 1), (0, 2), (0, 3), (0, 4)], Some(0)), ["CH₄ — methane", "CH₄ — carbon tetrahydride", "CH₄ — methane", "CH₄"]),
            (classify(&[6, 8, 8], &[(0, 1), (0, 2)], Some(0)), ["CO₂ — carbon dioxide", "CO₂ — carbon dioxide", "CO₂ — carbon dioxide", "CO₂"]),
            (classify(&[1, 9], &[(0, 1)], Some(0)), ["HF — hydrogen fluoride", "HF — hydrogen fluoride", "HF — fluorane", "HF"]),
            (classify(&[1, 8, 8, 1], &[(0, 1), (1, 2), (2, 3)], Some(0)), ["H₂O₂ — hydrogen peroxide", "H₂O₂ — dihydrogen dioxide", "H₂O₂ — dioxidane", "H₂O₂"]),
            (classify(&[8, 8], &[(0, 1)], Some(0)), ["O₂ — dioxygen", "O₂ — dioxygen", "O₂ — dioxygen", "O₂"]),
            (classify(&[1, 1], &[(0, 1)], Some(0)), ["H₂ — dihydrogen", "H₂ — dihydrogen", "H₂ — dihydrogen", "H₂"]),
            (classify(&[3, 1], &[(0, 1)], Some(0)), ["LiH — lithium hydride", "LiH — lithium hydride", "LiH — lithium hydride", "LiH"]),
            // a heteronuclear ion: IUPAC's name is additive (hydridohelium(1+)) — out of scope, the formula alone
            (classify(&[2, 1], &[(0, 1)], Some(1)), ["HeH⁺ — helium hydride cation", "HeH⁺", "HeH⁺", "HeH⁺"]),
            (classify_graph(&[1, 1, 1], &tri, true, Some(1)), ["H₃⁺ — trihydrogen cation", "H₃⁺ — trihydrogen(1+)", "H₃⁺ — trihydrogen(1+)", "H₃⁺"]),
            (classify(&[8, 1, 1, 1], &[(0, 1), (0, 2), (0, 3)], Some(1)), ["H₃O⁺ — hydronium", "H₃O⁺", "H₃O⁺ — oxidanium", "H₃O⁺"]),
            (classify(&[6, 8], &[(0, 1)], Some(0)), ["CO — carbon monoxide", "CO — carbon monoxide", "CO — carbon monoxide", "CO"]),
            (classify(&[7, 7, 8], &[(0, 1), (1, 2)], Some(0)), ["N₂O — dinitrogen monoxide (systematic)", "N₂O — dinitrogen monoxide", "N₂O — dinitrogen monoxide", "N₂O"]),
        ];
        for (n, want) in &cases {
            for (style, w) in [Common, Stoichiometric, Hydride, FormulaOnly].iter().zip(want) {
                assert_eq!(n.label_in(*style), *w, "{style:?}");
            }
        }
        // the remaining prefixes and stems, the ions, a lone atom, a ternary compound
        assert_eq!(stoichiometric_name(&[(1, 3), (5, 1)], 0).as_deref(), Some("boron trihydride"));
        assert_eq!(stoichiometric_name(&[(8, 2), (9, 2)], 0).as_deref(), Some("dioxygen difluoride")); // O before F (Table VI)
        assert_eq!(stoichiometric_name(&[(8, 2)], -2).as_deref(), Some("dioxide(2−)"));
        assert_eq!(stoichiometric_name(&[(1, 1)], -1).as_deref(), Some("hydride(1−)"));
        assert_eq!(stoichiometric_name(&[(1, 1)], 0).as_deref(), Some("monohydrogen"));
        // a lone cation has no "mono" (IR-5.3.2.2)
        assert_eq!(stoichiometric_name(&[(2, 1)], 1).as_deref(), Some("helium(1+)"));
        assert_eq!(stoichiometric_name(&[(1, 1)], 1).as_deref(), Some("hydrogen(1+)"));
        // Common's structural fallback and the stoichiometric style share the binary builder
        let n2o = classify(&[7, 7, 8], &[(0, 1), (1, 2)], Some(0));
        assert_eq!(n2o.text, n2o.stoichiometric);
        // a chain hydride name needs its backbone: C₂H₆ bridged only through H, H₂O₂ without O–O
        let bridged = classify(&[6, 1, 6, 1, 1, 1, 1, 1], &[(0, 1), (1, 2), (0, 3), (0, 4), (2, 5), (2, 6), (2, 7)], Some(0));
        assert_eq!(bridged.hydride, None);
        assert_eq!(bridged.label_in(Hydride), "C₂H₆ — dicarbon hexahydride"); // the stoichiometric fallback
        let no_oo = classify(&[1, 8, 1, 8], &[(0, 1), (1, 2), (2, 3)], Some(0)); // H–O–H–O, no O–O bond
        assert_eq!(no_oo.hydride, None);
        assert_eq!(no_oo.label_in(Hydride), "H₂O₂ — dihydrogen dioxide");
        assert_eq!(hydride_name(&[1, 8, 8, 1], &[(0, 1), (1, 2), (2, 3)], 0).as_deref(), Some("dioxidane"));
        assert_eq!(hydride_name(&[1, 7, 7, 1, 1, 1], &[(0, 1), (1, 2), (2, 3), (1, 4), (2, 5)], 0).as_deref(), Some("diazane"));
        assert_eq!(hydride_name(&[1, 7, 1, 7, 1, 1], &[(0, 1), (1, 2), (2, 3), (3, 4), (3, 5)], 0), None);
        assert_eq!(classify(&[1], &[], Some(0)).label_in(Stoichiometric), "H — monohydrogen");
        assert_eq!(stoichiometric_name(&[(1, 4), (6, 1), (8, 1)], 0), None);
        assert_eq!(classify(&[6, 8, 1, 1, 1, 1], &[(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)], Some(0)).label_in(Stoichiometric), "CH₄O");
        // the tooltip names the common name when the style differs, and the ring
        let w = &cases[0].0;
        assert_eq!(w.tooltip_in(Stoichiometric), "H₂O — dihydrogen monoxide\ncommon name: H₂O — water");
        assert_eq!(w.tooltip_in(Common), "H₂O — water");
        assert_eq!(cases[10].0.tooltip_in(Hydride), "H₃⁺ — trihydrogen(1+) (ring)\ncommon name: H₃⁺ — trihydrogen cation");
        assert_eq!(NameStyle::default(), Common);
    }

    #[test]
    fn labels_follow_the_ui_copy() {
        assert_eq!(classify(&[8, 1, 1], &[(0, 1), (0, 2)], Some(0)).label(), "H₂O — water");
        assert_eq!(classify(&[3, 1], &[(0, 1)], Some(0)).label(), "LiH — lithium hydride");
        assert_eq!(classify(&[2, 1, 1], &[(0, 1), (0, 2)], Some(0)).label(), "HeH₂ — helium dihydride (systematic)");
        assert_eq!(classify(&[1], &[], Some(0)).label(), "H");
        assert_eq!(classify(&[1], &[], Some(0)).tooltip(), "H");
    }
}
