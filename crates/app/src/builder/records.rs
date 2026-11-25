//! The builder's molecule-shape database (spec `docs/superpowers/specs/2026-11-25-builder-database-design.md`
//! §5): the records, the shipped data file's format and PubChem's V2000 SDF.
//!
//! **Rule 1 (spec §3, signed off by the user 2026-11-25).** Like [`super::chem`]'s tables, a record is
//! a starting-geometry aid: it decides where the builder places nuclei on Insert / Replace and what
//! a search puts on the canvas, nothing else. No value from it leaves `crates/app/src/builder/`.
//!
//! **The data file** (`crates/app/data/molecules.txt`, embedded by [`Library::shipped`]) is plain
//! text, one keyword per line, so a change reads as a diff. The first line is [`FORMAT`]; blank lines
//! and `#` comments are skipped; each record is
//!
//! ```text
//! mol ethanol                       the name
//! syn ethyl alcohol                 a synonym (any number of lines)
//! source pubchem 702                or: source lira functional=LSDA quality=Accurate ionic=accepted
//!                                       multiplicity=1 max_force=0.0003 commit=1a2b3c4d
//! stereo no                         yes: the shape is one stereoisomer of the graph
//! atom O 0 -1.1712 0.2997 0 0.87 -0.5    symbol, formal charge, x y z (Å), canvas x y
//! bond 0 1 1                        atom indices (from 0) and the order
//! end
//! ```
//!
//! Numbers are written with Rust's shortest round-trip `Display`, so `parse(write(v)) == v` for
//! every record that [`Record::validate`]s.
use crate::ui::ptable::Z_ENABLED;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;
use std::sync::OnceLock;
use units::elements::{symbol, z_of_symbol};

/// The data file's version line.
pub const FORMAT: &str = "ASMOLECULES 1";

/// The shipped data file, embedded (spec §5: ≤ 3 MB).
pub const DATA: &[u8] = include_bytes!("../../data/molecules.txt");

/// The most atoms a record may have, hydrogens included (spec §5).
pub const MAX_ATOMS: usize = 30;

/// The range every bonded pair's 3D distance must lie in (Å, spec §5's integrity check).
pub const BOND_RANGE: (f64, f64) = (0.5, 3.5);

/// Where a record's 3D shape comes from (the footer's label, spec §4 Goal 5).
#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    /// PubChem's computed 3D conformer (PubChem3D, MMFF94s-optimised) of compound `cid`.
    PubChem { cid: u64 },
    /// Our own Relax at the production defaults (spec §5.1): the functional, the quality preset, the
    /// scene's ionic-policy tag, the multiplicity the solve settled on, the final verified max |F|
    /// (Hartree/Bohr) and the commit the generator ran at.
    Lira { functional: String, quality: String, ionic: String, multiplicity: u8, max_force: f64, commit: String },
}

/// One molecule: its graph (elements, formal charges, bonds with orders), its 3D shape and a 2D
/// layout for the canvas. Atom `i` is the same atom in every per-atom list.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub name: String,
    pub synonyms: Vec<String>,
    pub source: Source,
    /// Atomic numbers.
    pub z: Vec<u8>,
    /// Formal charges.
    pub charge: Vec<i8>,
    /// `(a, b, order)`: atom indices from 0, order 1–3 (Kekulé).
    pub bonds: Vec<(usize, usize, u8)>,
    /// The 3D shape (Å).
    pub xyz: Vec<[f64; 3]>,
    /// The 2D layout in canvas units ([`canvas_layout`]: the mean bond is 1, y points down).
    pub xy: Vec<[f32; 2]>,
    /// The shape is one stereoisomer of a graph that has several (spec §6.4).
    pub stereo_defined: bool,
}

impl Record {
    /// The lookup key: the Hill formula ([`hill`]) and the net formal charge.
    pub fn formula_key(&self) -> (String, i32) {
        (hill(&self.z), self.charge.iter().map(|&q| q as i32).sum())
    }

    /// Spec §5's checks on one record: 1–[`MAX_ATOMS`] atoms of Z 1–[`Z_ENABLED`]; every per-atom
    /// list as long as `z`; finite coordinates; bonds of order 1–3 between two distinct atoms in
    /// range, none twice, each within [`BOND_RANGE`] in 3D; and names and source fields the format
    /// can hold.
    #[allow(dead_code)] // used outside tests by the online lookup's fetched records (builder database plan, Task 4)
    pub fn validate(&self) -> Result<(), String> {
        let n = self.z.len();
        if n == 0 {
            return Err("no atoms".into());
        }
        if n > MAX_ATOMS {
            return Err(format!("{n} atoms (at most {MAX_ATOMS})"));
        }
        if let Some(&z) = self.z.iter().find(|&&z| !(1..=Z_ENABLED).contains(&z)) {
            return Err(format!("Z {z} is outside 1–{Z_ENABLED}"));
        }
        for (what, len) in [("charges", self.charge.len()), ("3D positions", self.xyz.len()), ("2D positions", self.xy.len())] {
            if len != n {
                return Err(format!("{len} {what} for {n} atoms"));
            }
        }
        if self.xyz.iter().flatten().any(|c| !c.is_finite()) || self.xy.iter().flatten().any(|c| !c.is_finite()) {
            return Err("a coordinate is not finite".into());
        }
        let mut seen = HashSet::new();
        for &(a, b, order) in &self.bonds {
            if a >= n || b >= n {
                return Err(format!("bond {a}–{b}: an atom index is out of range ({n} atoms)"));
            }
            if a == b {
                return Err(format!("bond {a}–{b} joins an atom to itself"));
            }
            if !(1..=3).contains(&order) {
                return Err(format!("bond {a}–{b} has order {order} (1–3)"));
            }
            if !seen.insert((a.min(b), a.max(b))) {
                return Err(format!("bond {a}–{b} is listed twice"));
            }
            let d = dist(self.xyz[a], self.xyz[b]);
            if !(BOND_RANGE.0..=BOND_RANGE.1).contains(&d) {
                return Err(format!("bond {a}–{b} is {d:.3} Å long (outside {}–{} Å)", BOND_RANGE.0, BOND_RANGE.1));
            }
        }
        for s in std::iter::once(&self.name).chain(&self.synonyms) {
            if s.is_empty() || s.trim() != s || s.chars().any(char::is_control) {
                return Err(format!("the name {s:?} is empty, padded or has a control character"));
            }
        }
        if let Source::Lira { functional, quality, ionic, max_force, commit, .. } = &self.source {
            for s in [functional, quality, ionic, commit] {
                if s.is_empty() || s.chars().any(|c| c.is_whitespace() || c.is_control()) {
                    return Err(format!("the source field {s:?} is not one word"));
                }
            }
            if !max_force.is_finite() || *max_force < 0.0 {
                return Err(format!("max |F| {max_force} is not a finite non-negative number"));
            }
        }
        Ok(())
    }
}

/// The Hill formula of a list of atomic numbers ("C2H6O", "H4N", "HO"): with carbon, C then H then
/// the rest alphabetically; without, everything alphabetically. A count of 1 is left out.
pub fn hill(z: &[u8]) -> String {
    let mut count: BTreeMap<&str, usize> = BTreeMap::new();
    for &z in z {
        *count.entry(symbol(z)).or_default() += 1;
    }
    let carbon = count.contains_key("C");
    let first = ["C", "H"].into_iter().filter(|s| carbon && count.contains_key(s));
    let rest = count.keys().copied().filter(|s| !(carbon && (*s == "C" || *s == "H")));
    first
        .chain(rest)
        .map(|s| match count[s] {
            1 => s.to_string(),
            n => format!("{s}{n}"),
        })
        .collect()
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// The data file's text for `records` (the module doc's format). Exact for records that validate.
#[allow(dead_code)] // used outside tests by the online lookup's cache (builder database plan, Task 4)
pub fn write(records: &[Record]) -> String {
    let mut s = format!("{FORMAT}\n# the builder's molecule shapes; crates/app/src/builder/records.rs documents the format\n");
    for r in records {
        s += &format!("\nmol {}\n", r.name);
        for syn in &r.synonyms {
            s += &format!("syn {syn}\n");
        }
        s += &match &r.source {
            Source::PubChem { cid } => format!("source pubchem {cid}\n"),
            Source::Lira { functional, quality, ionic, multiplicity, max_force, commit } => format!(
                "source lira functional={functional} quality={quality} ionic={ionic} multiplicity={multiplicity} max_force={max_force} commit={commit}\n"
            ),
        };
        s += if r.stereo_defined { "stereo yes\n" } else { "stereo no\n" };
        for (i, &z) in r.z.iter().enumerate() {
            let (p, q) = (r.xyz[i], r.xy[i]);
            s += &format!("atom {} {} {} {} {} {} {}\n", symbol(z), r.charge[i], p[0], p[1], p[2], q[0], q[1]);
        }
        for &(a, b, order) in &r.bonds {
            s += &format!("bond {a} {b} {order}\n");
        }
        s += "end\n";
    }
    s
}

/// Reads the data file's text (the module doc's format). Structure only: the caller validates.
pub fn parse(text: &str) -> Result<Vec<Record>, String> {
    let mut lines = text.lines().enumerate().map(|(i, l)| (i + 1, l.trim()));
    match lines.next() {
        Some((_, l)) if l == FORMAT => {}
        Some((_, l)) => return Err(format!("line 1: {l:?} is not the version line {FORMAT:?}")),
        None => return Err("empty: no version line".into()),
    }
    let mut out = Vec::new();
    let mut cur: Option<Partial> = None;
    for (no, line) in lines {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, rest) = line.split_once(' ').map_or((line, ""), |(k, r)| (k, r.trim()));
        let at = |e: String| format!("line {no}: {e}");
        match (key, cur.as_mut()) {
            ("mol", None) => cur = Some(Partial::new(rest)),
            ("mol", Some(_)) => return Err(at("a record starts before the last one ended".into())),
            (_, None) => return Err(at(format!("{key:?} outside a record"))),
            ("end", Some(_)) => out.push(cur.take().unwrap().finish().map_err(at)?),
            (_, Some(p)) => p.line(key, rest).map_err(at)?,
        }
    }
    match cur {
        Some(p) => Err(format!("the record {:?} has no end line", p.name)),
        None => Ok(out),
    }
}

/// A record while its lines are read.
struct Partial {
    name: String,
    synonyms: Vec<String>,
    source: Option<Source>,
    stereo: Option<bool>,
    z: Vec<u8>,
    charge: Vec<i8>,
    xyz: Vec<[f64; 3]>,
    xy: Vec<[f32; 2]>,
    bonds: Vec<(usize, usize, u8)>,
}

impl Partial {
    fn new(name: &str) -> Partial {
        Partial { name: name.into(), synonyms: vec![], source: None, stereo: None, z: vec![], charge: vec![], xyz: vec![], xy: vec![], bonds: vec![] }
    }

    fn line(&mut self, key: &str, rest: &str) -> Result<(), String> {
        let words: Vec<&str> = rest.split_whitespace().collect();
        match key {
            "syn" => self.synonyms.push(rest.into()),
            "source" if self.source.is_some() => return Err("a second source line".into()),
            "source" => self.source = Some(parse_source(&words)?),
            "stereo" if self.stereo.is_some() => return Err("a second stereo line".into()),
            "stereo" => {
                self.stereo = Some(match rest {
                    "yes" => true,
                    "no" => false,
                    _ => return Err(format!("stereo {rest:?}: yes or no")),
                })
            }
            "atom" => {
                let [sym, q, x, y, z, u, v] = words[..] else { return Err(format!("atom: 7 fields, not {}", words.len())) };
                self.z.push(z_of_symbol(sym).ok_or_else(|| format!("atom: unknown element {sym:?}"))?);
                self.charge.push(num(q)?);
                self.xyz.push([num(x)?, num(y)?, num(z)?]);
                self.xy.push([num(u)?, num(v)?]);
            }
            "bond" => {
                let [a, b, order] = words[..] else { return Err(format!("bond: 3 fields, not {}", words.len())) };
                self.bonds.push((num(a)?, num(b)?, num(order)?));
            }
            _ => return Err(format!("unknown line {key:?}")),
        }
        Ok(())
    }

    fn finish(self) -> Result<Record, String> {
        Ok(Record {
            source: self.source.ok_or_else(|| format!("{:?} has no source line", self.name))?,
            stereo_defined: self.stereo.ok_or_else(|| format!("{:?} has no stereo line", self.name))?,
            name: self.name,
            synonyms: self.synonyms,
            z: self.z,
            charge: self.charge,
            bonds: self.bonds,
            xyz: self.xyz,
            xy: self.xy,
        })
    }
}

fn num<T: std::str::FromStr>(s: &str) -> Result<T, String> {
    s.parse().map_err(|_| format!("{s:?} is not a valid number here"))
}

/// `pubchem <cid>` or `lira functional=… quality=… ionic=… multiplicity=… max_force=… commit=…`.
fn parse_source(words: &[&str]) -> Result<Source, String> {
    const KEYS: [&str; 6] = ["functional", "quality", "ionic", "multiplicity", "max_force", "commit"];
    match words {
        ["pubchem", cid] => Ok(Source::PubChem { cid: num(cid)? }),
        ["lira", fields @ ..] => {
            if fields.len() != KEYS.len() {
                return Err(format!("source lira: {} fields, not {}", KEYS.len(), fields.len()));
            }
            let mut v = [""; 6];
            for (i, (f, key)) in fields.iter().zip(KEYS).enumerate() {
                v[i] = f.strip_prefix(key).and_then(|r| r.strip_prefix('=')).ok_or_else(|| format!("source lira: {f:?} where {key}=… belongs"))?;
            }
            Ok(Source::Lira {
                functional: v[0].into(),
                quality: v[1].into(),
                ionic: v[2].into(),
                multiplicity: num(v[3])?,
                max_force: num(v[4])?,
                commit: v[5].into(),
            })
        }
        _ => Err(format!("source {:?}: pubchem <cid> or lira …", words.join(" "))),
    }
}

/// The records, grouped by [`Record::formula_key`] for the lookup (spec §6.3).
pub struct Library {
    /// Sorted by key, stably: within a key, the order they were given in.
    records: Vec<Record>,
    by_key: HashMap<(String, i32), Range<usize>>,
}

impl Library {
    /// The shipped list ([`DATA`]), parsed once. A file that doesn't parse is a build defect (the
    /// `records` tests parse and validate every entry), so it panics.
    pub fn shipped() -> &'static Library {
        static LIB: OnceLock<Library> = OnceLock::new();
        LIB.get_or_init(|| {
            let text = std::str::from_utf8(DATA).expect("molecules.txt is UTF-8");
            Library::from_records(parse(text).unwrap_or_else(|e| panic!("molecules.txt: {e}")))
        })
    }

    pub fn from_records(mut v: Vec<Record>) -> Library {
        v.sort_by_cached_key(Record::formula_key);
        let mut by_key: HashMap<(String, i32), Range<usize>> = HashMap::new();
        for (i, r) in v.iter().enumerate() {
            by_key.entry(r.formula_key()).and_modify(|span| span.end = i + 1).or_insert(i..i + 1);
        }
        Library { records: v, by_key }
    }

    /// The records with this key, in their given order; none for an unknown key.
    pub fn candidates(&self, key: &(String, i32)) -> &[Record] {
        self.by_key.get(key).map_or(&[], |span| &self.records[span.clone()])
    }

    /// Every record, grouped by key.
    #[allow(dead_code)] // used outside tests by name search (builder database plan, Task 5)
    pub fn records(&self) -> &[Record] {
        &self.records
    }
}

/// Reads the first molecule of a V2000 SDF (PubChem's `SDF?record_type=3d|2d`): the atomic numbers,
/// the formal charges, the bonds (atoms from 0, order 1–3) and the coordinates (Å). The charges come
/// from the `M  CHG` lines when there are any, which then supersede the atom block's charge codes
/// (the V2000 rule), and otherwise from those codes. V3000, and aromatic (4) or query bond types,
/// are refused.
#[allow(dead_code)] // used outside tests by the online lookup (builder database plan, Task 4)
pub fn parse_sdf(sdf: &str) -> Result<(Vec<u8>, Vec<i8>, Vec<(usize, usize, u8)>, Vec<[f64; 3]>), String> {
    let lines: Vec<&str> = sdf.lines().collect();
    let counts = lines.get(3).ok_or("SDF: no counts line")?;
    if !counts.contains("V2000") {
        return Err(format!("SDF: {counts:?} is not a V2000 counts line"));
    }
    let na: usize = num(col(counts, 0, 3)).map_err(|e| format!("SDF atom count: {e}"))?;
    let nb: usize = num(col(counts, 3, 6)).map_err(|e| format!("SDF bond count: {e}"))?;
    let block = |i: usize| lines.get(i).copied().ok_or_else(|| format!("SDF: ends at line {}", i + 1));
    let (mut z, mut charge, mut xyz) = (Vec::with_capacity(na), Vec::with_capacity(na), Vec::with_capacity(na));
    for i in 4..4 + na {
        let l = block(i)?;
        let sym = col(l, 31, 34);
        z.push(z_of_symbol(sym).ok_or_else(|| format!("SDF line {}: unknown element {sym:?}", i + 1))?);
        xyz.push([num(col(l, 0, 10))?, num(col(l, 10, 20))?, num(col(l, 20, 30))?]);
        charge.push(match col(l, 36, 39) {
            "" | "0" | "4" => 0, // 4: a doublet radical, uncharged
            "1" => 3,
            "2" => 2,
            "3" => 1,
            "5" => -1,
            "6" => -2,
            "7" => -3,
            c => return Err(format!("SDF line {}: charge code {c:?}", i + 1)),
        });
    }
    let mut bonds = Vec::with_capacity(nb);
    for i in 4 + na..4 + na + nb {
        let l = block(i)?;
        let index = |s: &str| -> Result<usize, String> {
            match num::<usize>(s)? {
                k if (1..=na).contains(&k) => Ok(k - 1),
                k => Err(format!("SDF line {}: atom {k} of {na}", i + 1)),
            }
        };
        let order: u8 = num(col(l, 6, 9))?;
        if !(1..=3).contains(&order) {
            return Err(format!("SDF line {}: bond type {order} (only 1–3; aromatic and query bonds are refused)", i + 1));
        }
        bonds.push((index(col(l, 0, 3))?, index(col(l, 3, 6))?, order));
    }
    let mut chg_seen = false;
    for l in lines[4 + na + nb..].iter().take_while(|l| !l.starts_with("M  END")) {
        let Some(rest) = l.strip_prefix("M  CHG") else { continue };
        if !chg_seen {
            charge.iter_mut().for_each(|q| *q = 0);
            chg_seen = true;
        }
        let w: Vec<&str> = rest.split_whitespace().collect();
        let n: usize = num(w.first().copied().unwrap_or(""))?;
        if w.len() != 1 + 2 * n {
            return Err(format!("SDF: {l:?} lists {n} charges but has {} fields", w.len() - 1));
        }
        for pair in w[1..].chunks(2) {
            let a: usize = num(pair[0])?;
            if !(1..=na).contains(&a) {
                return Err(format!("SDF: {l:?} names atom {a} of {na}"));
            }
            charge[a - 1] = num(pair[1])?;
        }
    }
    Ok((z, charge, bonds, xyz))
}

/// A fixed-width V2000 field: columns `a..b`, trimmed ("" past the line's end).
fn col(line: &str, a: usize, b: usize) -> &str {
    line.get(a..b.min(line.len())).unwrap_or("").trim()
}

/// The canvas layout rule (shared by the search's stamp, the shipped list's generator and the LIRA
/// entries): a 2D layout's x and y (Å, y up as in an SDF; z is ignored), centred on its centroid,
/// y flipped (the canvas's y points down) and scaled so the mean bond is 1 canvas unit (unscaled
/// with no bonds).
#[allow(dead_code)] // used outside tests by the online lookup (builder database plan, Task 4)
pub fn canvas_layout(p: &[[f64; 3]], bonds: &[(usize, usize, u8)]) -> Vec<[f32; 2]> {
    if p.is_empty() {
        return Vec::new();
    }
    let n = p.len() as f64;
    let (cx, cy) = (p.iter().map(|q| q[0]).sum::<f64>() / n, p.iter().map(|q| q[1]).sum::<f64>() / n);
    let mean = bonds.iter().map(|&(a, b, _)| (p[a][0] - p[b][0]).hypot(p[a][1] - p[b][1])).sum::<f64>() / bonds.len().max(1) as f64;
    let s = if mean > 1e-9 { 1.0 / mean } else { 1.0 };
    // + 0.0 turns −0 into 0, so the data file never reads "-0"
    p.iter().map(|q| [((q[0] - cx) * s + 0.0) as f32, ((cy - q[1]) * s + 0.0) as f32]).collect()
}

#[cfg(test)]
mod tests {
    //! The seed `crates/app/data/molecules.txt` (Task 1; Task 6 replaces it) was made from PubChem's
    //! 3D and 2D SDFs, fetched once by hand on 2026-11-25:
    //! `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/<cid>/SDF?record_type=3d` and
    //! `…?record_type=2d` for CIDs 962 (water), 297 (methane), 702 (ethanol), 241 (benzene),
    //! 223 (ammonium) and 961 (hydroxide; PubChem has a 3D record for it). Each 2D graph matched its
    //! 3D graph atom for atom; `xyz` is the 3D SDF's, `xy` is [`canvas_layout`] of the 2D SDF's.
    use super::*;

    /// A water record built by hand (PubChem CID 962's 3D shape), the base the refusals mutate.
    fn water() -> Record {
        Record {
            name: "water".into(),
            synonyms: vec![],
            source: Source::PubChem { cid: 962 },
            z: vec![8, 1, 1],
            charge: vec![0, 0, 0],
            bonds: vec![(0, 1, 1), (0, 2, 1)],
            xyz: vec![[0.0, 0.0, 0.0], [0.2774, 0.8929, 0.2544], [0.6068, -0.2383, -0.7169]],
            xy: vec![[0.0, 0.0], [0.86, -0.5], [-0.86, -0.5]],
            stereo_defined: false,
        }
    }

    const WATER_SDF: &str = "962
  -OEChem-11252601033D

  3  2  0     0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.2774    0.8929    0.2544 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.6068   -0.2383   -0.7169 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
M  END
> <PUBCHEM_COMPOUND_CID>
962

$$$$
";

    /// PubChem CID 223's block, its atom-block charge field set to 0 so only `M  CHG` carries the +1.
    const AMMONIUM_SDF: &str = "223
  -OEChem-11252601033D

  5  4  0     0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    0.9645    0.2796    0.2138 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0075   -0.8860   -0.5188 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5235   -0.1202    0.8751 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.4486    0.7266   -0.5701 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
M  CHG  1   1   1
M  END
$$$$
";

    #[test]
    fn the_seed_file_parses_and_validates() {
        let lib = Library::shipped();
        assert!(lib.records().len() >= 6);
        for r in lib.records() {
            r.validate().unwrap_or_else(|e| panic!("{}: {e}", r.name));
        }
        for name in ["water", "methane", "ethanol", "benzene", "ammonium", "hydroxide"] {
            assert!(lib.records().iter().any(|r| r.name == name), "{name} is missing");
        }
    }

    #[test]
    fn the_format_round_trips() {
        let v = Library::shipped().records().to_vec();
        assert_eq!(parse(&write(&v)).unwrap(), v);
        // a Lira source and synonyms, which the seed may not carry
        let mut w = water();
        w.synonyms = vec!["oxidane".into(), "dihydrogen monoxide".into()];
        w.stereo_defined = true;
        w.source = Source::Lira {
            functional: "LSDA".into(),
            quality: "Accurate".into(),
            ionic: "accepted".into(),
            multiplicity: 2,
            max_force: 3.25e-4,
            commit: "seed".into(),
        };
        assert_eq!(parse(&write(&[w.clone(), water()])).unwrap(), vec![w, water()]);
    }

    #[test]
    fn ammonium_and_hydroxide_carry_their_charges() {
        let nh4 = Library::shipped().records().iter().find(|r| r.name == "ammonium").unwrap();
        assert_eq!(nh4.formula_key(), ("H4N".into(), 1));
        let oh = Library::shipped().records().iter().find(|r| r.name == "hydroxide").unwrap();
        assert_eq!(oh.formula_key(), ("HO".into(), -1));
    }

    #[test]
    fn formula_keys_follow_hill_order() {
        let lib = Library::shipped();
        let key = |n: &str| lib.records().iter().find(|r| r.name == n).unwrap().formula_key();
        assert_eq!(key("water"), ("H2O".into(), 0));
        assert_eq!(key("methane"), ("CH4".into(), 0));
        assert_eq!(key("ethanol"), ("C2H6O".into(), 0));
        assert_eq!(key("benzene"), ("C6H6".into(), 0));
        // candidates come back by key
        let c = lib.candidates(&("C2H6O".into(), 0));
        assert!(c.iter().any(|r| r.name == "ethanol"));
        assert!(c.iter().all(|r| r.formula_key() == ("C2H6O".into(), 0)));
        assert!(lib.candidates(&("C2H6O".into(), 1)).is_empty());
    }

    #[test]
    fn validate_refuses_out_of_range_entries() {
        assert_eq!(water().validate(), Ok(()));
        let mut r = water();
        r.z[0] = 19;
        assert!(r.validate().is_err(), "Z 19");
        let mut r = water();
        for i in 0..28 {
            if r.z.len() == 30 {
                assert_eq!(r.validate(), Ok(()), "30 atoms are allowed");
            }
            r.z.push(1);
            r.charge.push(0);
            r.xyz.push([10.0 + 2.0 * i as f64, 0.0, 0.0]);
            r.xy.push([0.0, 0.0]);
        }
        assert_eq!(r.z.len(), 31);
        assert!(r.validate().is_err(), "31 atoms");
        let mut r = water();
        r.xyz[1] = [4.0, 0.0, 0.0];
        assert!(r.validate().is_err(), "a 4 Å bond");
        let mut r = water();
        r.bonds.push((0, 99, 1));
        assert!(r.validate().is_err(), "a bond to atom 99");
        let mut r = water();
        r.bonds.push((1, 0, 1));
        assert!(r.validate().is_err(), "a duplicated bond");
        let mut r = water();
        r.bonds[0].2 = 4;
        assert!(r.validate().is_err(), "bond order 4");
        let mut r = water();
        r.xy.pop();
        assert!(r.validate().is_err(), "a short 2D layout");
        let mut r = water();
        r.xyz.pop();
        assert!(r.validate().is_err(), "a short 3D shape");
    }

    #[test]
    fn parse_refuses_a_wrong_version_and_names_the_line() {
        assert!(parse("ASMOLECULES 2\n").is_err());
        assert!(parse("").is_err());
        let bad = write(&[water()]).replace("atom H", "atom Xx");
        let e = parse(&bad).unwrap_err();
        assert!(e.contains("line"), "{e}");
    }

    #[test]
    fn sdf_v2000_water_parses() {
        let (z, q, bonds, xyz) = parse_sdf(WATER_SDF).unwrap();
        assert_eq!(z, vec![8, 1, 1]);
        assert_eq!(q, vec![0, 0, 0]);
        assert_eq!(bonds, vec![(0, 1, 1), (0, 2, 1)]);
        assert_eq!(xyz, vec![[0.0, 0.0, 0.0], [0.2774, 0.8929, 0.2544], [0.6068, -0.2383, -0.7169]]);
    }

    #[test]
    fn sdf_charges_come_from_the_m_chg_line() {
        let (z, q, bonds, _) = parse_sdf(AMMONIUM_SDF).unwrap();
        assert_eq!(z, vec![7, 1, 1, 1, 1]);
        assert_eq!(q, vec![1, 0, 0, 0, 0]);
        assert_eq!(bonds.len(), 4);
        // without an M  CHG line the atom block's code stands (5 = −1); with one, the block is ignored
        let block = AMMONIUM_SDF.replace("M  CHG  1   1   1\n", "");
        let block = block.replacen(" N   0  0", " N   0  5", 1);
        assert_eq!(parse_sdf(&block).unwrap().1, vec![-1, 0, 0, 0, 0]);
        let both = AMMONIUM_SDF.replacen(" N   0  0", " N   0  5", 1);
        assert_eq!(parse_sdf(&both).unwrap().1, vec![1, 0, 0, 0, 0]);
    }

    #[test]
    fn sdf_refuses_v3000_and_aromatic_bonds() {
        assert!(parse_sdf(&WATER_SDF.replace("V2000", "V3000")).is_err());
        assert!(parse_sdf(&WATER_SDF.replacen("  1  2  1  0", "  1  2  4  0", 1)).is_err());
        assert!(parse_sdf("").is_err());
    }

    #[test]
    fn the_2d_layout_has_unit_mean_bonds_and_canvas_y_down() {
        let xy = canvas_layout(&[[2.5369, -0.155, 0.0], [3.0739, 0.155, 0.0], [2.0, 0.155, 0.0]], &[(0, 1, 1), (0, 2, 1)]);
        let len = |a: [f32; 2], b: [f32; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
        assert!((0.5 * (len(xy[0], xy[1]) + len(xy[0], xy[2])) - 1.0).abs() < 1e-6);
        // the SDF's O sits below its Hs (smaller y, y up); on the canvas (y down) it stays below
        // them, so its y is the larger
        assert!(xy[0][1] > xy[1][1] && xy[0][1] > xy[2][1]);
        for r in Library::shipped().records() {
            let mean = r.bonds.iter().map(|&(a, b, _)| len(r.xy[a], r.xy[b])).sum::<f32>() / r.bonds.len() as f32;
            assert!((mean - 1.0).abs() < 1e-6, "{}: mean drawn bond {mean}", r.name);
        }
    }
}
