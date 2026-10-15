//! M7 generator gates per element (spec §6.2) and the pinned outputs table (review S6a).
//! Run: `cargo test -p validate --release --test pseudo_gen`. When the records move (a universal
//! constant changed, or Task 5/11/12 add a column or element), the test writes the new block to the
//! gitignored `crates/validate/last-run-generation-records.md` and fails; after physics review, copy it
//! between the markers in `references.md` ("M7 generation records").
//! A moved record must move `recipe_hash` (plan review M6, Global Constraints): when a row present in
//! both blocks differs and the hash line did not change, the failure says "bump LEDGER.version".
use qm::radial::{generate, record::records_table, LEDGER};
use std::collections::BTreeMap;

const BEGIN: &str = "<!-- m7-generation-records:begin -->";
const END: &str = "<!-- m7-generation-records:end -->";
/// Every Z the milestone covers (Task 4 review N9: the bump rule then guards all of them): Li, Na, Mg pinned as
/// refused by the partition, Be and Al by T5 at this recipe version; H and He are provisional until Task 11's
/// dimer gate; Al–Ar are enabled in the app by Task 12.
const ZS: [u8; 18] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];

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
    // T8 (Task 5): the filter ladder behind each h_req — (a) ΔE, Δε, (b) T3, (c) clamped, (d) F_SR/F_NL/F_pc — reported
    // (the block pins h_req only)
    for (z, r) in &rows {
        if let Ok(g) = r {
            let ladder: Vec<String> = g.filter.iter().map(|f| format!("h {:.2}: ΔE {:+.2e}, Δε {:.2e}, T3 {:.2e}, clamped {}, F {:.2e}/{:.2e}/{:.2e}{}", f.h, f.de, f.max_eig, f.t3_max,
                f.clamped, f.f_table[0], f.f_table[1], f.f_table[2], if f.passes() { "" } else { " (fails)" })).collect();
            eprintln!("Z = {z}: h_req {:?}; {}", g.h_req, ladder.join("; "));
        }
    }
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

/// Spec §6.2 (as revised by the Task 4 physics review): every gate on B–Ne (T1–T7 inside `generate`; here
/// their numbers are printed and the acceptance asserted against `LEDGER.gates`, never against literals); Li
/// refused by the partition (ONCV), Be refused by T5 at every ladder entry (pinned, like Li).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_gen")]
fn b_to_ne_pass_every_gate_li_and_be_are_refused() {
    assert!(generate(3).is_err_and(|e| e.to_string().contains("ONCV")));
    let be = generate(4).expect_err("Be is refused at this recipe version");
    eprintln!("Z = 4: {be}");
    assert!(be.to_string().contains("T5"), "Be's refusal names T5: {be}");
    let gates = &LEDGER.gates;
    for z in 5u8..=10 {
        let t = generate(z).unwrap_or_else(|e| panic!("Z = {z} refused: {e}"));
        let g = &t.gates;
        eprintln!("Z = {z}: f {} {:?} T1 {:.2e} T2 {:.1e} T3 {:?} T4 {:.2e} T5 {:?} T5 far {:?} T6 {:?}", t.f, t.local, g.t1, g.t2,
                  g.t3.iter().map(|r| (r.name.clone(), r.de_ae - r.de_pp)).collect::<Vec<_>>(), g.t4, g.t5, g.t5_far, g.t6);
        assert!(g.t1 < gates.t1_eig && g.t2 < gates.t2_norm && g.t4 < gates.t4_eig);
        assert!(g.t3.iter().all(|r| (r.de_ae - r.de_pp).abs() < gates.t3_delta_e));
        assert!(g.t5.iter().all(|(_, d)| *d < gates.t5_de_equiv));
        assert!(g.t6.iter().all(|(_, _, a, p)| (a - p).abs() < gates.t6_hardness));
    }
}
