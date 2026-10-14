//! M7 generator gates per element (spec §6.2) and the pinned outputs table (review S6a).
//! Run: `cargo test -p validate --release --test pseudo_gen`. When the records move (a universal
//! constant changed, or Task 5/11/12 add a column or element), the test writes the new block to the
//! gitignored `crates/validate/last-run-generation-records.md` and fails; after physics review, copy it
//! between the markers in `references.md` ("M7 generation records").
//! A moved record must move `recipe_hash` (plan review M6, Global Constraints): when a row present in
//! both blocks differs and the hash line did not change, the failure says "bump LEDGER.version".
use qm::radial::{generate, record::records_table};
use std::collections::BTreeMap;

const BEGIN: &str = "<!-- m7-generation-records:begin -->";
const END: &str = "<!-- m7-generation-records:end -->";
/// Li is listed to pin its refusal; Be–Ne are the elements the spec expects accepted (§6.2).
const ZS: [u8; 8] = [3, 4, 5, 6, 7, 8, 9, 10];

/// The block's first line, "recipe_hash = 0x… (version)".
fn hash_line(block: &str) -> &str { block.lines().next().unwrap_or("").trim() }

/// The table rows keyed by their Z cell (the header and separator rows skipped).
fn rows_by_z(block: &str) -> BTreeMap<String, String> {
    block.lines().filter(|l| l.starts_with("| ") && !l.starts_with("| Z ") && !l.starts_with("|---"))
        .filter_map(|l| l.split('|').nth(1).map(|z| (z.trim().to_string(), l.trim().to_string()))).collect()
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_gen")]
fn generation_record_table_matches_references_md() {
    let rows: Vec<_> = ZS.iter().map(|&z| (z, generate(z).map(|t| t.record))).collect();
    let table = records_table(&rows);
    // line endings normalised: references.md is checked out with CRLF where git converts (Windows,
    // core.autocrlf), `records_table` writes LF
    let refs = include_str!("../references.md").replace("\r\n", "\n");
    let pinned = refs.split(BEGIN).nth(1).and_then(|s| s.split(END).next()).map(str::trim).unwrap_or("");
    if pinned != table.trim() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/last-run-generation-records.md");
        std::fs::write(path, &table).expect("write the regenerated block");
        let (old, new) = (rows_by_z(pinned), rows_by_z(&table));
        let moved: Vec<&String> = old.iter().filter(|(z, row)| new.get(*z).is_some_and(|n| n != *row)).map(|(z, _)| z).collect();
        if !pinned.is_empty() && !moved.is_empty() && hash_line(pinned) == hash_line(&table) {
            panic!("the generation records of Z = {moved:?} moved but recipe_hash did not: bump LEDGER.version (plan Global Constraints, plan review M6), then regenerate — {path}:\n{table}");
        }
        panic!("the generation records differ from references.md — review, then copy {path} between the markers:\n{table}");
    }
}

/// Spec §6.2: every gate on Be–Ne (T1–T7 inside `generate`; here their numbers are printed and the
/// spec's acceptance asserted); Li refused with its reason.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_gen")]
fn be_to_ne_pass_every_gate_and_li_is_refused() {
    assert!(generate(3).is_err_and(|e| e.to_string().contains("ONCV")));
    for z in 4u8..=10 {
        let t = generate(z).unwrap_or_else(|e| panic!("Z = {z} refused: {e}"));
        let g = &t.gates;
        eprintln!("Z = {z}: f {} {:?} T1 {:.2e} T2 {:.1e} T3 {:?} T4 {:.2e} T5 {:?} T6 {:?}", t.f, t.local, g.t1, g.t2,
                  g.t3.iter().map(|r| (r.name.clone(), r.de_ae - r.de_pp)).collect::<Vec<_>>(), g.t4, g.t5, g.t6);
        assert!(g.t1 < 1e-3 && g.t2 < 1e-10 && g.t4 < 5e-3);
        assert!(g.t3.iter().all(|r| (r.de_ae - r.de_pp).abs() < 5e-3));
        assert!(g.t5.iter().all(|(_, d)| *d < 20e-3));
        assert!(g.t6.iter().all(|(_, _, a, p)| (a - p).abs() < 10e-3));
    }
}
