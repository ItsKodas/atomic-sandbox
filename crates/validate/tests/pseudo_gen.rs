//! M7 generator gates per element (spec §6.2) and the pinned outputs table (review S6a).
//! Run: `cargo test -p validate --release --test pseudo_gen`. When the records move (a universal
//! constant changed, or Task 5/11/12 add a column or element), the test writes the new block to the
//! gitignored `crates/validate/last-run-generation-records.md` and fails; after physics review, copy it
//! between the markers in `references.md` ("M7 generation records").
//! A moved record must move `recipe_hash` (plan review M6, Global Constraints): when a row present in
//! both blocks differs and the hash line did not change, the failure says "bump LEDGER.version".
//! `nlcc_crossover_is_the_minimax_rung` is the NLCC crossover ladder's table and selection rule (spec §4.5, §4.9; NLCC
//! ruling, 2026-10-16): it regenerates every Z at every rung c and asserts that `LEDGER.nlcc_crossover` is the minimax
//! rung.
use qm::radial::{generate, generate_at_crossover, record::records_table, GenerationRecord, Refusal, LEDGER};
use std::collections::BTreeMap;

const BEGIN: &str = "<!-- m7-generation-records:begin -->";
const END: &str = "<!-- m7-generation-records:end -->";
/// Every Z the milestone covers (Task 4 review N9: the bump rule then guards all of them): Li, Na, Mg pinned as
/// refused by the partition, Be and Al by T5 at this recipe version; H and He are generated here without the dimer
/// gate (CPU, no 3-D solves), so their rows read "provisional (dimer gate not run)" (Task 11, plan review M6) — the
/// gated H/He the providers use are recorded by `pseudo_oracles` and references.md "M7 `All` (Task 11)"; Al–Ar are
/// enabled in the app by Task 12. (The plan's Task 11 wrote ZS = 1–10; Task 12 had already extended it to 18.)
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
            let ladder: Vec<String> = g.filter.iter().map(|f| format!("h {:.2}: ΔE {:+.2e}, Δε {:.2e}, T3 {:.2e}, clamped {}, F {:.2e}/{:.2e}/{:.2e} (e: F_pc,nl {:.2e}, F_atom,nl {:.2e}){}", f.h, f.de, f.max_eig, f.t3_max,
                f.clamped, f.f_table[0], f.f_table[1], f.f_table[2], f.f_pc_nl, f.f_atom_nl(), if f.passes() { "" } else { " (fails)" })).collect();
            eprintln!("Z = {z}: h_req {:?}, h_eb {:?}; {}", g.h_req, g.h_eb, ladder.join("; "));
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
            panic!("the generation records of Z = {moved:?} moved but recipe_hash did not: bump LEDGER.version, or `DIMER_GATE` if only the dimer-gated species moved (plan Global Constraints, plan review M6; Task 11 \"two identities\"), then regenerate — {path}:\n{table}");
        }
        panic!("the generation records differ from references.md — review, then copy {path} between the markers:\n{table}");
    }
}

/// Spec §8 row 11: H and He pass the atomic gates (the explicit no-core branch: Z_v = Z, no NLCC); their rows are in the
/// pinned records. Generated here without the dimer gate, they are provisional (plan review M6): usable by no setup —
/// `IonicSetup::build` refuses them with `IonicError::Provisional`, at any h, before the h_req test.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_gen")]
fn hydrogen_and_helium_pass_the_atomic_gates() {
    for z in [1u8, 2] {
        let t = generate(z).unwrap_or_else(|e| panic!("Z = {z}: {e}"));
        assert_eq!((t.n_core, t.z_v), (0, z as f64));
        assert!(t.partial_core.r_nlc.is_none() && t.partial_core.r_core == 0.0);
        assert!(t.provisional && t.record.dimer.starts_with("provisional"), "Z = {z}: {}", t.record.dimer);
        let e = qm::IonicSetup::build(qm::ionic::IonicPolicy::All, 0.15, 0.15, &[z], &mut |_| Ok(false),
            &mut |z| Ok(std::sync::Arc::new(qm::radial::filter::filtered(&generate(z)?, 0.15)))).unwrap_err();
        assert!(matches!(e, qm::ionic::IonicError::Provisional { .. }), "Z = {z}: {e}");
        eprintln!("Z = {z}: f {}, {}, h_req {:?}, gates {:?}; refused by build: {e}", t.f, t.record.local, t.record.h_req, t.gates);
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

/// Spec §8 row 12: Al–Ar pass every gate (n = 2 core; the 3d unbound and dropped, spec §4.2 item 3) or are
/// refused with a reason; Na and Mg are refused with the two-s-state reason naming ONCV (spec §4.2).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_gen")]
fn al_through_ar_pass_gates_or_are_refused_with_reason() {
    for z in [11u8, 12] {
        let e = qm::radial::generate(z).expect_err("Na and Mg carry a semicore n = 2 shell");
        assert!(e.to_string().contains("two valence s states") && e.to_string().contains("ONCV"), "Z = {z}: {e}");
    }
    for z in 13u8..=18 {
        match qm::radial::generate(z) {
            Ok(t) => {
                assert_eq!(t.n_core, 10, "Z = {z}: the n = 2 shell is the core");
                assert_eq!(t.partition.dropped, vec![2], "Z = {z}: the 3d is unbound in the neutral LSDA atom (spec §4.2 item 3)");
                assert!(t.partition.channels.iter().all(|c| c.occupied), "Z = {z}: no unoccupied channel");
                // reported, not gated (T5 excludes an ungenerated l; rule 7): the d-wave scattering of the local
                // potential against the AE atom, δE_equiv on T5's window and mesh at r_d (`delta_e_equiv` with no
                // l = 2 projector is the local potential's); pinned in references.md's generation records as
                // "d scattering (ungated)"
                let gates = &qm::radial::LEDGER.gates;
                let e_min = t.partition.channels.iter().map(|c| c.eps).fold(f64::INFINITY, f64::min) - gates.t5_window;
                let e_max = t.partition.channels.iter().map(|c| c.eps).fold(f64::NEG_INFINITY, f64::max) + gates.t5_window;
                let steps = ((e_max - e_min) / gates.t5_mesh).round() as usize;
                let d_worst = (0..=steps).map(|k| qm::radial::gates::delta_e_equiv(&t, 2, e_min + k as f64 * gates.t5_mesh).abs()).fold(0.0, f64::max);
                eprintln!("Z = {z}: d scattering (ungated): max |δE_equiv| {:.2} mHa", d_worst * 1e3);
                eprintln!("Z = {z}: f {}, {}, h_req {:?}", t.f, t.record.local, t.record.h_req);
            }
            Err(e) => {
                let m = e.to_string();
                assert!(!m.is_empty() && (m.contains("ladder") || m.contains("valence")), "Z = {z}: a refusal must name its reason: {m}");
                eprintln!("Z = {z} refused: {m}");
            }
        }
    }
}

/// Review NOTE (M7 plan review): the Δ_core rule is "universal, but close" for Al — its 2p (core) lies only a few
/// tenths of a Hartree below the promotion threshold ε_3s − Δ_core. The margin is measured, not assumed: the 2p must
/// stay below the threshold (it stays core) and the 2s/2p of Na and Mg must be above it (they are promoted and the
/// element is refused). The numbers are printed for references.md ("Z = 11–18 enablement").
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_gen")]
fn delta_core_margins_of_na_mg_al_are_reported() {
    use qm::radial::{ground_state, SpinMode};
    let delta = LEDGER.delta_core;
    for z in 11u8..=13 {
        let a = ground_state(z, SpinMode::Restricted).expect("the reference atom converges");
        let eps = |n: u8, l: u8| a.orbitals.iter().find(|o| o.n == n && o.l == l && o.occ[0] + o.occ[1] > 0.0).map(|o| o.eps[0]);
        let (e2p, e3s) = (eps(2, 1).expect("2p occupied"), eps(3, 0).expect("3s occupied"));
        let threshold = e3s - delta;
        // positive: the 2p lies below the threshold (stays core); negative: it is within Δ_core (promoted)
        let margin = threshold - e2p;
        eprintln!("Z = {z}: ε_2p {e2p:.6}, ε_3s {e3s:.6}, threshold ε_3s − Δ_core = {threshold:.6}, margin {margin:+.6} Ha");
        assert_eq!(margin > 0.0, z == 13, "Z = {z}: only Al keeps the n = 2 shell as core (margin {margin:+.4} Ha)");
    }
}

/// One generated element at a rung: its record and the unfiltered ρ_pc(0) = AB (e/Bohr³).
type Rung = Result<(GenerationRecord, f64), Refusal>;

/// One element's lines of the ladder table: f, local, r_nlc, ρ_pc(0), T1–T7 (every T3 Δ, Hund included), T8 (a)–(d) and
/// (e) per ladder h, h_req and h_eb.
fn ladder_lines(c: f64, z: u8, r: &Rung) -> String {
    let el = units::elements::symbol(z);
    let (g, rho_pc0) = match r { Ok(x) => x, Err(e) => return format!("c = {c} Z = {z} {el}: refused: {e}") };
    let gt = &g.gates;
    let t3: Vec<String> = gt.t3.iter().map(|x| format!("{} {:+.3}", x.name, (x.de_pp - x.de_ae) * 1e3)).collect();
    let t5: Vec<String> = gt.t5.iter().map(|(l, d)| format!("l{l} {:.3}", d * 1e3)).collect();
    let t6 = gt.t6.iter().map(|(_, _, a, p)| (a - p).abs()).fold(0.0f64, f64::max);
    let t7 = (gt.t7.iter().filter(|r| r.n_kb == r.ae.len()).count(), gt.t7.len());
    let mut s = format!("c = {c} Z = {z} {el}: f {} {}, r_nlc {}, ρ_pc(0) {rho_pc0:.4}; T1 {:.2e} T2 {:.1e} T3 [{}] mHa T4 {:.3} mHa T5 [{}] mHa T6 {:.2} mHa/e T7 {}/{}; h_req {:?}, h_eb {:?}",
                        g.f, g.local, g.r_nlc.map_or("—".into(), |x| format!("{x:.4}")), gt.t1, gt.t2, t3.join(", "), gt.t4 * 1e3, t5.join(", "), t6 * 1e3, t7.0, t7.1,
                        g.h_req, g.h_eb);
    for f in &g.filter {
        s += &format!("\n    h {:.2}: (a) ΔE {:+.3} mHa Δε {:.3} mHa (b) T3 {:.3} mHa (c) clamped {} (d) F_SR {:.2e} F_NL {:.2e} F_pc {:.2e} Σ {:.2e} → {}; (e) F_pc,nl {:.2e} F_atom,nl {:.2e} → {}",
                      f.h, f.de * 1e3, f.max_eig * 1e3, f.t3_max * 1e3, f.clamped, f.f_table[0], f.f_table[1], f.f_table[2], f.f_table.iter().sum::<f64>(),
                      if f.passes() { "pass" } else { "fail" }, f.f_pc_nl, f.f_atom_nl(), if f.eggbox_certified() { "certified" } else { "not certified" });
    }
    s
}

/// Every Z of `ZS` generated at crossover c, 6 threads (each generate is single-threaded, radial, ≲ 100 MB).
fn generate_all(c: f64) -> Vec<(u8, Rung)> {
    let out: Vec<std::sync::Mutex<Option<Rung>>> = ZS.iter().map(|_| std::sync::Mutex::new(None)).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..6 {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= ZS.len() { break; }
                *out[i].lock().unwrap() = Some(generate_at_crossover(ZS[i], c).map(|t| (t.record.clone(), t.partial_core.a * t.partial_core.b)));
            });
        }
    });
    ZS.iter().zip(out).map(|(&z, m)| (z, m.into_inner().unwrap().expect("every Z ran"))).collect()
}

/// The rungs confirmed in 3-D (NLCC ruling, Ruling 5 step 2: the lone-O per-term attribution at h = 0.20 and 0.15, checks
/// a and b, `pseudo.rs` `lone_o_attribution_confirms_the_crossover_gpu`; c = 1 is LFC's own criterion, the baseline).
/// ½ was not tried (the ruling: trying it needs a new ruling); ⅛ is inadmissible (it refuses N).
const CONFIRMED_3D: &[f64] = &[1.0, 0.25];

/// Spec §4.5 "The rule", §4.9 (NLCC ruling, 2026-10-16, Ruling 3): every Z = 1–18 generated at every rung c of
/// `LEDGER.nlcc_crossover_ladder` (CPU, radial, 6 threads), the table printed (`ladder_lines`), and the minimax rule:
/// - admissible: every element accepted at c = 1 is accepted at c (T1–T7), and c is in `CONFIRMED_3D`;
/// - M(c) = max over Z and over ladder h ≤ h_req^∪(Z) of F_atom,nl(Z, c, h) (T8(e)), h_req^∪(Z) the coarsest h_req any
///   T1–T7-admissible rung gives Z (the union, so no rung improves its score by retreating to a finer h_req);
/// - the recipe's c is the admissible argmin of M.
/// Asserted: the -3 baseline (c = 1 reproduces the -3 records' h_req: (d) is linear again), and argmin = `LEDGER.nlcc_crossover`.
/// Printed: M(c) for every T1–T7-admissible rung, confirmed or not, and the T3 Hund cost per rung.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_gen")]
fn nlcc_crossover_is_the_minimax_rung() {
    let t0 = std::time::Instant::now();
    let rungs: Vec<(f64, Vec<(u8, Rung)>)> = LEDGER.nlcc_crossover_ladder.iter().map(|&c| (c, generate_all(c))).collect();
    let base = &rungs[0].1;
    let pinned_m3: BTreeMap<u8, Option<f64>> = [(1, Some(0.35)), (2, Some(0.20)), (5, Some(0.35)), (6, Some(0.30)), (7, Some(0.30)), (8, Some(0.20)), (9, Some(0.15)), (10, None),
                                                 (14, Some(0.35)), (15, Some(0.35)), (16, Some(0.35)), (17, Some(0.30)), (18, Some(0.25))].into_iter().collect();
    let base_hreq: BTreeMap<u8, Option<f64>> = base.iter().filter_map(|(z, r)| r.as_ref().ok().map(|(g, _)| (*z, g.h_req))).collect();
    assert_eq!(base_hreq, pinned_m3, "c = 1 must reproduce the -3 records' h_req (T8(d) linear)");
    // T1–T7 admissibility: every element accepted at c = 1 is accepted at c
    let accepted = |rows: &[(u8, Rung)]| -> Vec<u8> { rows.iter().filter(|(_, r)| r.is_ok()).map(|(z, _)| *z).collect() };
    let base_acc = accepted(base);
    let t17: Vec<&(f64, Vec<(u8, Rung)>)> = rungs.iter().filter(|(_, rows)| { let a = accepted(rows); base_acc.iter().all(|z| a.contains(z)) }).collect();
    for (c, rows) in &rungs {
        for (z, r) in rows { eprintln!("{}", ladder_lines(*c, *z, r)); }
        let hund: Vec<String> = rows.iter().filter_map(|(z, r)| r.as_ref().ok().and_then(|(g, _)| g.gates.t3.iter().find(|x| x.name == "Hund")
            .map(|x| format!("{} {:+.3}", units::elements::symbol(*z), (x.de_pp - x.de_ae) * 1e3)))).collect();
        let hs: Vec<String> = rows.iter().map(|(z, r)| format!("{} {}", units::elements::symbol(*z), match r {
            Ok((g, _)) => format!("{}/{}", g.h_req.map_or("—".into(), |h| format!("{h:.2}")), g.h_eb.map_or("—".into(), |h| format!("{h:.2}"))), Err(_) => "refused".into() })).collect();
        eprintln!("c = {c}: h_req/h_eb [{}]; T3 Hund Δ (mHa) [{}]; T1–T7 admissible: {}", hs.join(", "), hund.join(", "), t17.iter().any(|(x, _)| x == c));
    }
    // h_req^∪(Z): the coarsest h_req over the T1–T7-admissible rungs
    let mut union: BTreeMap<u8, f64> = BTreeMap::new();
    for (_, rows) in &t17 { for (z, r) in rows { if let Ok((g, _)) = r { if let Some(h) = g.h_req { let e = union.entry(*z).or_insert(h); *e = e.max(h); } } } }
    let m_of = |rows: &[(u8, Rung)]| -> (f64, String) {
        let mut worst = (0.0f64, String::new());
        for (z, r) in rows {
            let (Ok((g, _)), Some(&hu)) = (r, union.get(z)) else { continue };
            for f in g.filter.iter().filter(|f| f.h <= hu + 1e-12) {
                if f.f_atom_nl() > worst.0 { worst = (f.f_atom_nl(), format!("{} at h = {:.2}", units::elements::symbol(*z), f.h)); }
            }
        }
        worst
    };
    let mut best: Option<(f64, f64)> = None;
    for (c, rows) in &t17 {
        let (m, at) = m_of(rows);
        let confirmed = CONFIRMED_3D.contains(c);
        eprintln!("M(c = {c}) = {m:.3e} ({at}){}", if confirmed { "" } else { " — not confirmed in 3-D, not eligible" });
        if confirmed && best.is_none_or(|(_, bm)| m < bm) { best = Some((*c, m)); }
    }
    eprintln!("h_req^∪ {union:?}; the minimax rung {best:?}; LEDGER.nlcc_crossover = {}; {:.0} s", LEDGER.nlcc_crossover, t0.elapsed().as_secs_f64());
    assert_eq!(best.map(|b| b.0), Some(LEDGER.nlcc_crossover), "the ledger's crossover is not the minimax rung");
}
