//! The recipe's universal constants — spec §4.9, "the complete rule-1 ledger" — as ONE `const`
//! table. Every *recipe* constant the generator (`qm::radial`) and its consumers (the filter,
//! `qm::ionic`, the dimer gate) use — every number that can move a generated table — is a field of
//! [`LEDGER`]; no other file under `crates/qm/src/radial/` declares a numeric constant
//! (`gate_thresholds_are_constants`, Task 4). The method-convergence literals that cannot move a
//! converged result beyond its own tolerance (bisection width, WKB end, overflow rescale, DIIS
//! regularisation, …) stay inline and are listed in the module doc of the file that uses them
//! (`numerov`, `atom`; review S2, 2026-10-15). Nothing is indexed by Z. A change to
//! any value is a recipe change: every element is regenerated and re-gated, and `recipe_hash`
//! (Task 4: FNV-1a 64 of this table's `Debug` rendering, which includes `version` and `xc`) changes,
//! invalidating cached tables and surrogate provenances.
//!
//! `numerics` holds the solver knobs the spec names in §4.1/§4.6 (radial SCF tolerance, DIIS) and the
//! plan's G2 zero-count mesh (Spec question 5): they do not change a converged result beyond their
//! tolerance, but they are hashed too.
//!
//! # Two identities (M7 Task 11; the controller's provenance requirement)
//! The dimer gate on f (spec §4.4) decides the tables of Z ≤ `dimer_z_max` only. Switching it on must invalidate everything
//! made under the ungated H and He (plan review M6(b)) and nothing else: a surface that pseudises only species above the
//! bound (every `CoresOnly` surface — H and He have no core) uses byte-identical tables before and after. So the gate is
//! not a bump of `version` (which would have orphaned every saved `CoresOnly` model — the user's water models among them)
//! but its own identity [`DIMER_GATE`]: [`recipe_hash`] = ledger + gate (the cache, the records, `IonicSetup`);
//! [`core_recipe_hash`] = the ledger alone (the identity of every table above the bound); [`species_recipe_hash`] picks per
//! Z (the generation record's hash). The surrogate's provenance tag hashes the identity of the species its policy can
//! pseudise (`providers::pseudo::surface_hash`).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Xc { LsdaPw92 }

/// Nonlinear core correction mode: `Lfc` (Louie–Froyen–Cohen, spec §4.5); `Off` is the switch the
/// precision milestone's QMC variant needs (spec §3, "Future requirement").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NlccMode { Lfc, Off }

/// Local-channel candidates in the universal preference order (spec §4.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalCandidate { HighestChannel, Polynomial }

/// Gate thresholds (spec §4.6, §4.8). Energies in Hartree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gates {
    pub t1_eig: f64, pub t2_norm: f64, pub t3_delta_e: f64, pub t4_eig: f64,
    /// T5 (spec §4.8, Task 4 review revision): gated at max |δE_equiv| < `t5_de_equiv` (= T4's eigenvalue
    /// tolerance) over ε_l ± `t5_window` per channel; max over ε_l ± `t5_report_window` reported only.
    pub t5_de_equiv: f64, pub t5_window: f64, pub t5_report_window: f64, pub t5_mesh: f64, pub t5_rd_factor: f64,
    pub t6_hardness: f64, pub t6_dq: f64,
    pub g2_eig: f64, pub g2_window: f64,
    pub h_req_tol: f64,
    /// T8(d) (spec §4.8, §5.1 "Translation invariance"; Task 5 physics review): per-atom table egg-box force
    /// F_SR + F_NL + F_pc at the §6.6 offsets {±h/4}³, Ha/Bohr = (§6.6 gate 3e-5)/3 shared over the ≤ 5 atoms of M7's molecules.
    pub h_req_force: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Numerics {
    pub scf_tol: f64, pub scf_max_iter: usize, pub diis_history: usize, pub mixing: f64, pub g2_mesh: f64,
    /// Consecutive β-halvings of the linear-mixing fallback before an SCF step that unbinds an
    /// occupied shell is reported as genuinely unbound (review M1: β_min = mixing/2^6 ≈ 5e-3).
    pub backtrack_max: usize,
    /// Size of the King-Smith–Payne–Lin band-completion Chebyshev basis on (q_c, π/h] (Task 5's
    /// `filter_radial`; plan review Q3 ruling).
    pub kspl_terms: usize,
    /// The band completion's outer C^∞ step spans [γ_r R, γ_r R + kspl_step_over_h·h]: the smallest of {1, 2, 4, 8}
    /// that brings every accepted element under T8(d) at its h_req (Task 5 physics review, Table A: C and N at 0.30 need 8).
    pub kspl_step_over_h: f64,
    /// 3-D sampling tables are uniform in d with spacing h / table_points_per_h, filled exactly; 4-point Lagrange,
    /// relative error ≤ (9/16)(π/table_points_per_h)⁴/24 = 5.3e-10 (Bernstein; spec §5.1 "Sampling").
    pub table_points_per_h: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ledger {
    /// The recipe version of every table. History: -2 T5 per channel (Task 4 review), -3 the filter/T8 (Task 5), -4 the
    /// NLCC crossover c = ¼ (NLCC ruling). The dimer gate on f (Task 11) is NOT a bump of this string: it can move only the
    /// tables of Z ≤ `dimer_z_max`, so it has its own identity, [`DIMER_GATE`], hashed into [`recipe_hash`] and into the
    /// gated species' [`species_recipe_hash`] only (module doc, "Two identities").
    pub version: &'static str,
    /// Radial log grid r_i = r0·e^{iδ} up to r_max (Bohr) — §4.1.
    pub r0: f64, pub r_max: f64, pub delta: f64,
    /// Ruling S9's multiplicity cap (`sim::sampling::allowed_multiplicities`; a `sim` test pins them equal).
    pub multiplicity_max: u32,
    /// Semicore promotion window and occupation-scan window (Ha) — §4.2, §4.1.
    pub delta_core: f64, pub delta_scan: f64,
    /// l_max = max occupied valence l + this — §4.2.
    pub l_max_above_valence: u8,
    /// The r_c ladder, softest first — §4.4.
    pub f_ladder: [f64; 4],
    pub local_order: [LocalCandidate; 2],
    pub nlcc: NlccMode,
    /// The NLCC crossover ladder (§4.5, §4.9; O₂ egg-box study ruling, 2026-10-16): r_nlc where ρ_c = c·ρ_v, c from
    /// this universal ladder, largest first; c = 1 is LFC's own criterion. A smaller c lowers the amplitude of LFC's C¹
    /// junction, J ∝ ρ_c(r_nlc), which sets ρ_pc's spectrum near the grid's band edge.
    pub nlcc_crossover_ladder: [f64; 4],
    /// The one global c of the ladder the recipe uses (NLCC ruling, 2026-10-16, Ruling 3): among the admissible rungs —
    /// every element accepted at c = 1 stays accepted (T1–T7) and the rung is confirmed in 3-D — the minimiser of
    /// M(c) = max_Z max_{h ≤ h_req^∪(Z)} F_atom,nl(Z, c, h) (T8(e)), h_req^∪ the coarsest h_req any admissible rung gives Z.
    /// Checked by `validate --test pseudo_gen` `nlcc_crossover_is_the_minimax_rung`; never per element.
    pub nlcc_crossover: f64,
    /// TM polynomial degree (7 coefficients) — §4.3.
    pub tm_degree: u8,
    /// r_cmp: where |V_ion + Z_v/r| < this (Ha) for good — §4.3.
    pub v_ion_tail: f64,
    pub gates: Gates,
    /// q_c = this · π/h; the outer step starts at gamma_r · R (R the table's support) — §5.1.
    pub q_c_over_nyquist: f64, pub gamma_r: f64,
    /// σ_LR = this · h — §5.1.
    pub sigma_lr_over_h: f64,
    /// V_SR's support R_SR = max(r_cmp, r_floor), Z_v erfc(r_floor/a)/r_floor = this (Ha); also ρ_pc's support (S1) — §5.1.
    pub v_tail_floor: f64,
    /// The h ladder for h_req (Bohr), coarsest first — §4.8 T8.
    pub h_ladder: [f64; 5],
    /// r_core encloses all but this fraction of the core electrons — §5.6.
    pub eps_core: f64,
    /// A shell's r_peak (spec §4.4: r_c = f·r_peak) is its outermost local maximum of |u| above this
    /// fraction of max|u| — recipe-affecting, hence here (review S2).
    pub r_peak_floor: f64,
    /// The dimer gate on f applies for Z ≤ dimer_z_max: R_e within dimer_re (Bohr), ω within dimer_omega_rel — §4.4, §6.4.
    pub dimer_z_max: u8, pub dimer_re: f64, pub dimer_omega_rel: f64,
    /// The dimer gate's bond-length bracket, in units of the element's outermost valence r_peak
    /// (Task 11; Spec question 9): a bound AE dimer must have its minimum inside it.
    pub dimer_bracket: [f64; 2],
    pub xc: Xc,
    pub numerics: Numerics,
}

pub const LEDGER: Ledger = Ledger {
    version: "m7-tm-kb-nlcc-4",
    r0: 1e-5, r_max: 60.0, delta: 0.008,
    multiplicity_max: 5,
    delta_core: 2.0, delta_scan: 0.1,
    l_max_above_valence: 1,
    f_ladder: [1.6, 1.4, 1.2, 1.0],
    local_order: [LocalCandidate::HighestChannel, LocalCandidate::Polynomial],
    nlcc: NlccMode::Lfc, nlcc_crossover_ladder: [1.0, 0.5, 0.25, 0.125], nlcc_crossover: 0.25,
    tm_degree: 12,
    v_ion_tail: 1e-8,
    gates: Gates {
        t1_eig: 1e-3, t2_norm: 1e-10, t3_delta_e: 5e-3, t4_eig: 5e-3,
        t5_de_equiv: 5e-3, t5_window: 0.5, t5_report_window: 1.0, t5_mesh: 0.01, t5_rd_factor: 1.2,
        t6_hardness: 10e-3, t6_dq: 0.25,
        g2_eig: 1e-3, g2_window: 0.5,
        h_req_tol: 1e-3, h_req_force: 2e-6,
    },
    q_c_over_nyquist: 0.7, gamma_r: 1.5,
    sigma_lr_over_h: 2.0,
    v_tail_floor: 1e-6,
    h_ladder: [0.35, 0.30, 0.25, 0.20, 0.15],
    eps_core: 1e-3,
    r_peak_floor: 1e-3,
    dimer_z_max: 2, dimer_re: 0.02, dimer_omega_rel: 0.05, dimer_bracket: [1.0, 4.0],
    xc: Xc::LsdaPw92,
    numerics: Numerics { scf_tol: 1e-10, scf_max_iter: 400, diis_history: 6, mixing: 0.3, g2_mesh: 1e-3, backtrack_max: 6, kspl_terms: 24,
                         kspl_step_over_h: 8.0, table_points_per_h: 256 },
};

/// The identity of the dimer gate on f (spec §4.4; M7 Task 11, plan review M6): the code that decides the f of every
/// Z ≤ `LEDGER.dimer_z_max` by the homonuclear dimer against its own all-electron σ → 0 dimer. A change to that code (or to
/// what "gated" means) changes this string, and with it the identity of the gated species' tables only.
pub const DIMER_GATE: &str = "dimer-gate-1: grid dimer at the finest ladder h, AE sigma -> 0 per R, bisection in dimer_bracket*r_peak";

/// FNV-1a 64 over the table's `Debug` rendering (f64 `Debug` is the shortest round-trip decimal, so
/// the rendering — and the hash — is stable across builds and platforms). Every field, `version` and
/// `xc` included, enters the rendering.
pub fn hash_ledger(l: &Ledger) -> u64 { fnv(0xcbf2_9ce4_8422_2325, format!("{l:?}").as_bytes()) }

fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes { h ^= b as u64; h = h.wrapping_mul(0x0100_0000_01b3); }
    h
}

/// The recipe's identity (spec §4.9) — "two identities", module doc: the ledger and [`DIMER_GATE`]. It keys the
/// generation cache and the pinned records; it moved when the dimer gate was switched on (Task 11: every table, number
/// and model made under the ungated H is invalid by it).
pub fn recipe_hash() -> u64 { fnv(hash_ledger(&LEDGER), format!("|{DIMER_GATE}").as_bytes()) }

/// The identity of the tables of every Z above `LEDGER.dimer_z_max` (the dimer gate never runs for them): the ledger
/// alone. At `m7-tm-kb-nlcc-4` it equals the -4 recipe hash 0xeef0a918057a9fbe, so a surface that pseudises only such
/// species (every `CoresOnly` surface: H and He have no core) keeps its identity across Task 11.
pub fn core_recipe_hash() -> u64 { hash_ledger(&LEDGER) }

/// One hash, one potential per element (spec §4.9, §11 M6): Z's table identity — [`recipe_hash`] for a dimer-gated
/// species, [`core_recipe_hash`] for every other.
pub fn species_recipe_hash(z: u8) -> u64 { if z <= LEDGER.dimer_z_max { recipe_hash() } else { core_recipe_hash() } }

#[cfg(test)]
mod tests {
    use super::*;

    /// Spec §8 row 4: the ledger of §4.9 is one const table and nothing in `radial/` is indexed by Z.
    /// Grep-style: in every other radial source file's library code (before `#[cfg(test)]`, comments
    /// skipped) (1) no numeric `const`/`static` declaration; (2) no Z dispatch — `match z`, `z` as an
    /// index, or `z` compared with anything — unless the same line reads a `LEDGER.` field (the dimer
    /// gate's Z ≤ LEDGER.dimer_z_max is a universal rule). Task 5 added "filter.rs".
    #[test]
    fn gate_thresholds_are_constants() {
        const FILES: &[(&str, &str)] = &[
            ("mod.rs", include_str!("mod.rs")), ("grid.rs", include_str!("grid.rs")), ("numerov.rs", include_str!("numerov.rs")),
            ("atom.rs", include_str!("atom.rs")), ("partition.rs", include_str!("partition.rs")), ("tm.rs", include_str!("tm.rs")),
            ("kb.rs", include_str!("kb.rs")), ("recipe.rs", include_str!("recipe.rs")), ("nlcc.rs", include_str!("nlcc.rs")),
            ("gates.rs", include_str!("gates.rs")), ("record.rs", include_str!("record.rs")), ("filter.rs", include_str!("filter.rs")),
        ];
        const NUMERIC: [&str; 11] = [": f64", ": f32", ": u8", ": u16", ": u32", ": u64", ": usize", ": i32", ": i64", ": [f64", ": [u8"];
        let ident = |c: char| c.is_alphanumeric() || c == '_';
        for (name, src) in FILES {
            let lib = src.split("#[cfg(test)]").next().unwrap();
            for (k, line) in lib.lines().enumerate() {
                let t = line.trim_start();
                if t.starts_with("//") { continue; }
                let decl = ["const ", "pub const ", "pub(crate) const ", "static ", "pub static "].iter().any(|p| t.starts_with(p));
                assert!(!(decl && NUMERIC.iter().any(|n| t.contains(n))), "{name}:{}: a numeric constant outside the ledger: {t}", k + 1);
                if t.contains("LEDGER.") { continue; }
                let chars: Vec<char> = t.chars().collect();
                for (i, &c) in chars.iter().enumerate() {
                    if c != 'z' || (i > 0 && ident(chars[i - 1])) || chars.get(i + 1).is_some_and(|&n| ident(n)) { continue; }
                    let before: String = chars[..i].iter().collect();
                    let after: String = chars[i + 1..].iter().collect::<String>().trim_start().to_string();
                    let compared = ["==", "!=", "<", ">"].iter().any(|op| after.starts_with(op))
                        || ["== ", "!= ", "< ", "> ", "<= ", ">= "].iter().any(|op| before.trim_end().ends_with(op.trim_end()));
                    let indexed = before.trim_end().ends_with('[') || after.starts_with("as usize]");
                    let matched = before.trim_end().ends_with("match");
                    assert!(!(compared || indexed || matched), "{name}:{}: a Z dispatch (rule 1): {t}", k + 1);
                }
            }
        }
    }

    #[test]
    fn recipe_hash_tracks_every_switch() {
        let h = recipe_hash();
        assert_ne!(hash_ledger(&Ledger { nlcc: NlccMode::Off, ..LEDGER }), h, "the QMC switch (spec §3) must change the hash");
        assert_ne!(hash_ledger(&Ledger { gamma_r: 1.6, ..LEDGER }), h);
        assert_ne!(hash_ledger(&Ledger { version: "other", ..LEDGER }), h);
        assert_ne!(hash_ledger(&Ledger { f_ladder: [1.6, 1.4, 1.2, 0.9], ..LEDGER }), h);
        assert_eq!(recipe_hash(), h, "stable within a build");
        // the two identities (module doc): the gate is in the recipe's, not in the core tables'
        assert_ne!(recipe_hash(), core_recipe_hash());
        assert_eq!(core_recipe_hash(), hash_ledger(&LEDGER));
        assert_eq!((species_recipe_hash(1), species_recipe_hash(2), species_recipe_hash(8)), (recipe_hash(), recipe_hash(), core_recipe_hash()));
    }

    /// Task 11 (the controller's requirement): switching the dimer gate on did not change the identity of any table above
    /// the bound — the core hash is the -4 recipe hash pinned in references.md before Task 11 (records block
    /// "recipe_hash = 0xeef0a918057a9fbe (m7-tm-kb-nlcc-4)"), so every `CoresOnly` provenance from -4 is still valid.
    #[test]
    fn the_core_tables_keep_their_minus_4_identity() {
        assert_eq!(LEDGER.version, "m7-tm-kb-nlcc-4");
        assert_eq!(core_recipe_hash(), 0xeef0_a918_057a_9fbe, "the ledger moved: the core identity changed (a recipe change: re-pin, and expect every pseudo model to be refused)");
    }

    /// Spec §4.8 T5 (Task 4 physics review, rule-6 revision by derivation): δE_equiv is the first-order
    /// eigenvalue error of a channel-l state wholly inside r_d, so its threshold is T4's eigenvalue tolerance.
    /// Spec §4.9 (O₂ egg-box study ruling): the crossover is one global value taken from the universal ladder, which
    /// starts at LFC's own c = 1 and halves.
    #[test]
    fn nlcc_crossover_is_a_rung_of_the_ladder() {
        assert_eq!(LEDGER.nlcc_crossover_ladder[0], 1.0);
        assert!(LEDGER.nlcc_crossover_ladder.windows(2).all(|w| w[1] == 0.5 * w[0]));
        assert!(LEDGER.nlcc_crossover_ladder.contains(&LEDGER.nlcc_crossover));
        assert_ne!(hash_ledger(&Ledger { nlcc_crossover: 0.5, ..LEDGER }), hash_ledger(&Ledger { nlcc_crossover: 0.25, ..LEDGER }));
    }

    #[test]
    fn t5_threshold_is_the_eigenvalue_tolerance() {
        assert_eq!(LEDGER.gates.t5_de_equiv, LEDGER.gates.t4_eig);
    }
}
