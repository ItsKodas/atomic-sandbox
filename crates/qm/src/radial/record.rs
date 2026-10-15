//! The generation record (spec §4.4's pinned outputs table, review S6a; §4.9): what the recipe chose for
//! an element and the numbers its gates measured, under one `recipe_hash`. CPU, atomic units (Hartree,
//! Bohr). `records_table` is the block `crates/validate/tests/pseudo_gen.rs` pins in `references.md`
//! (no timings: they are reported, never pinned). Task 5 fills `h_req` and `filter`, Task 11 `dimer`.
use super::gates::GateReport;
use super::ledger::{recipe_hash, LEDGER};
use super::partition::{l_name, Refusal};
use super::recipe::PseudoTable;

/// One row of Task 5's filter ladder (spec §4.8 T8): h, the filtered radial atom's total-energy error, its worst
/// eigenvalue error and the worst filtered T3 difference (Ha), the XC-domain count and the table egg-box force.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FilterRow {
    pub h: f64, pub de: f64, pub max_eig: f64, pub t3_max: f64,
    /// T8(c): points where the XC clamp would engage, over the reference and every T3 configuration.
    pub clamped: usize,
    /// T8(d): [F_SR, F_NL, F_pc], Ha/Bohr.
    pub f_table: [f64; 3],
}

impl FilterRow {
    /// T8 (spec §4.8): (a) |ΔE|, max |Δε| < h_req_tol; (b) t3_max < t3_delta_e; (c) clamped = 0; (d) Σ f_table < h_req_force.
    /// `filter::h_req` and `ionic::check` both use this, so they cannot drift.
    pub fn passes(&self) -> bool {
        let g = &LEDGER.gates;
        self.de.abs() < g.h_req_tol && self.max_eig < g.h_req_tol && self.t3_max < g.t3_delta_e
            && self.clamped == 0 && self.f_table.iter().sum::<f64>() < g.h_req_force
    }
}

/// The record of one generated element. `extra_states`: (l, AE ε, KB root) for every bound state of the
/// reference beyond the channel's own below the G2 window top — the weak (n+1)s the short-range neutral
/// LSDA well binds (an LSDA artefact of the self-interaction-truncated potential; the real atom has a full
/// Rydberg series, far deeper — Task 3 review N8). `g2_missing`: AE states the pseudo-atom does not bind
/// (Task 3 review S1). `seconds`: the whole `generate` call, reported only.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenerationRecord {
    pub z: u8, pub recipe_hash: u64, pub f: f64, pub local: String, pub r_c: Vec<(u8, f64)>, pub z_v: f64, pub n_core: u32,
    pub e_kb: Vec<(u8, f64)>, pub gss: Vec<(u8, f64)>, pub r_nlc: Option<f64>, pub r_cmp: f64, pub r_core: f64,
    pub gates: GateReport, pub extra_states: Vec<(u8, f64, f64)>, pub g2_missing: Vec<(u8, Vec<f64>, Vec<f64>)>,
    pub h_req: Option<f64>, pub filter: Vec<FilterRow>, pub dimer: String, pub seconds: f64,
}

pub fn record_of(t: &PseudoTable, seconds: f64) -> GenerationRecord {
    let top = (t.partition.channels.iter().filter(|c| c.occupied).map(|c| c.eps).fold(f64::NEG_INFINITY, f64::max)
        + LEDGER.gates.g2_window).min(0.0);
    let mut extra_states = Vec::new();
    for (l, roots) in &t.ghost.g2_roots {
        let ae: Vec<f64> = t.partition.spectrum[*l as usize].iter().copied().filter(|&e| e < top).collect();
        let own = usize::from(t.partition.channels.iter().any(|c| c.l == *l));
        extra_states.extend(ae.iter().zip(roots).skip(own).map(|(a, k)| (*l, *a, *k)));
    }
    GenerationRecord {
        z: t.z, recipe_hash: recipe_hash(), f: t.f,
        local: match t.kb.local_l { Some(l) => format!("{}-local", l_name(l)), None => "polynomial".into() },
        r_c: t.channels.iter().map(|c| (c.l, c.r_c)).collect(), z_v: t.z_v, n_core: t.n_core,
        e_kb: t.kb.projectors.iter().map(|p| (p.l, p.e_kb)).collect(), gss: t.kb.projectors.iter().map(|p| (p.l, p.gss)).collect(),
        r_nlc: t.partial_core.r_nlc, r_cmp: t.kb.r_cmp, r_core: t.partial_core.r_core,
        gates: t.gates.clone(), extra_states, g2_missing: t.ghost.g2_missing.clone(), h_req: None, filter: Vec::new(), seconds,
        // Task 11's dimer_verdict overwrites this for Z ≤ dimer_z_max (pass / fail / not applicable / provisional)
        dimer: if t.z <= LEDGER.dimer_z_max { "not run".into() } else { "not applicable (Z above the dimer-gate bound)".into() },
    }
}

/// The pinned outputs table (review S6a): the hash line, then one row per Z; refusals name their reason.
/// No timings (reported elsewhere, never pinned).
pub fn records_table(rows: &[(u8, Result<GenerationRecord, Refusal>)]) -> String {
    let list = |v: &[(u8, f64)], d: usize| v.iter().map(|(l, x)| format!("{}: {x:.d$}", l_name(*l))).collect::<Vec<_>>().join(", ");
    let mut s = format!("recipe_hash = 0x{:016x} ({})\n\n", recipe_hash(), LEDGER.version);
    s += "| Z | el | f | local | r_c (Bohr) | Z_v | E_KB (Ha) | GSS (Ha) | r_nlc | r_core | h_req |\n|---|---|---|---|---|---|---|---|---|---|---|\n";
    for (z, r) in rows {
        let el = units::elements::symbol(*z);
        match r {
            Ok(g) => s += &format!("| {z} | {el} | {:.1} | {} | {} | {:.0} | {} | {} | {} | {:.3} | {} |\n",
                g.f, g.local, list(&g.r_c, 3), g.z_v, list(&g.e_kb, 4), list(&g.gss, 4),
                g.r_nlc.map_or("—".into(), |r| format!("{r:.3}")), g.r_core, g.h_req.map_or("—".into(), |h| format!("{h:.2}"))),
            // a refusal's reason in one cell: a '|' in it would split the row
            Err(e) => s += &format!("| {z} | {el} | refused | {} | | | | | | | |\n", e.to_string().replace('|', "¦")),
        }
    }
    s
}
