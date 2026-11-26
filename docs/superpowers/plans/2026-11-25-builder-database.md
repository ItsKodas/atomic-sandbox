# Molecule builder database shapes: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Insert / Replace in the 2D molecule builder place known molecules at database 3D shapes, from PubChem records or LIRA-relaxed entries. A search box stamps molecules onto the canvas. Unknown drawings keep today's embedding, labelled approximate.

**Architecture:** Five new files in `crates/app/src/builder/`:

| File | Holds |
|---|---|
| `records.rs` | the record type, the shipped data file and SDF parsing |
| `iso.rs` | fragments and exact atom mapping |
| `shape.rs` | the hand-off geometry and the source labels |
| `online.rs` | the SMILES writer, PubChem client, lookup thread and cache |
| `search.rs` | name search and stamping |

The data file `crates/app/data/molecules.txt` is embedded with `include_bytes!`, and two dev-only `#[ignore]`d generators produce it. Nothing outside `crates/app` changes, apart from docs.

**Tech stack:** Rust, egui (the existing builder window), `ureq` 2.12 (already a dependency for the updater), the `units` crate for Å → Bohr.

**Spec:** `docs/superpowers/specs/2026-11-25-builder-database-design.md` (approved by the user 2026-11-25). Every task's implementer reads the spec sections its task names.

**Model:** every task is **Opus** (memory "no Fable": all subagents run on Opus).

## Global constraints

- **Rule 1 (spec §3).** Records and LIRA shapes only place nuclei in the builder.
  - No value from them may reach `sim`, `qm`, `qm-gpu`, `providers`, `surrogate`, forces, energies, the bond detector or the names' graph.
  - Code lives in `crates/app/src/builder/` plus `crates/app/data/molecules.txt`.
  - `Handoff` keeps its fields and meaning: nuclei, charge, relax, grid, label.
- **Atomic units** inside: record coordinates are stored in Å and converted only with `units::angstrom_to_bohr`.
- **Elements:** Z ≤ 18 only (`ptable::Z_ENABLED`). Entries hold ≤ 30 atoms including H. The data file is ≤ 3 MB.
- **Network** (spec §7):
  - HTTPS to `pubchem.ncbi.nlm.nih.gov/rest/pug/` only, ≤ 5 requests per second.
  - 5 s timeout per lookup.
  - Lookups start after the drawing has been unchanged for 1 s.
  - Never on the UI thread.
  - **No request at all** when the setting is off, under `--replay` or `--screenshot`, or in any test (a failing test transport enforces it).
  - User agent `atomic-sandbox-builder/<version>`, generic, with no user data.
- **Cache:** `paths::user_root()/molecules/`. A corrupt file is skipped and refetched, never a crash.
- **Today's output is kept:**
  - a drawing with no matched fragment hands off **bit-identically** to today (`embed::embed` + `place_beside`);
  - **Relax after** defaults to on, unchanged;
  - builder hand-offs stay unreplayable (`replay::unreplayable_kind` → "builder").
- **Footer label strings (spec §4 Goal 5, verbatim):**
  - `shape: PubChem "<name>" (CID <n>)`
  - `shape: relaxed with LIRA (LSDA, Accurate)`
  - `shape: approximate (not in the database) — Relax after recommended`
  - `looking up online…`
  - `shape: approximate (offline)`
  - several fragments: `shape: <k> from PubChem, <j> relaxed with LIRA, <m> approximate`, with zero parts omitted
  - stereo addendum: `stereo as in the database (CID <n>)`
- **GPU:** the user is using the GPU. Until the coordinator says it is free:
  - only CPU tests run: `cargo test -p app --release --bin app -- builder::`, and `ui::` where named;
  - Task 7's generator and the full `cargo test -p app --release` wait.
  - Builds run at BelowNormal with `-j 4`.
- **Commits:**
  - Use pathspecs (`git commit -o -- <paths>`).
  - End every message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  - Never commit the main checkout's untracked `models/*.model`, `profiles/*.profile`, `scenes/h3+.scene` or `research_notes/`.

## Review focus

1. **A drawing edited while its lookup is in flight.** A late answer must not label or place the new drawing. Owned by Task 4: `a_late_answer_for_an_old_drawing_is_dropped`.
2. **Symmetric molecules with many equivalent mappings** (benzene, methane, SF₆-like centres), where the matcher might blow its step cap. These must still match well inside the cap. Owned by Task 2: `symmetric_molecules_match_inside_the_cap`.
3. **The same molecule as two fragments, or a molecule beside lone atoms.** Each fragment is matched on its own, and the ≥ 3 Bohr gap holds. Owned by Task 3: `two_waters_and_a_lone_atom_keep_their_gaps`.
4. **The box-fit check and the enlarge offer with database shapes.** The reach must be computed from the shape the hand-off will actually use, not from `embed.rs`. Owned by Task 3: `the_fit_check_uses_the_database_shape`.
5. **Unusual name queries:** a blank query, whitespace, a CID with leading zeros, a name with Unicode subscripts ("H₂O"), or a formula in lower case. None may panic or send a malformed URL. Owned by Task 5: `odd_queries_never_panic_or_send_malformed_urls`.

---

### Task 1: Records, the data file format and SDF parsing — **Opus**

**Files:**
- Create: `crates/app/src/builder/records.rs` (`pub mod records;` in `builder/mod.rs`)
- Create: `crates/app/data/molecules.txt`, holding a **seed** of 6 hand-checked entries for tests: water, methane, ethanol, benzene, ammonium (NH₄⁺) and hydroxide (OH⁻). It is replaced by Task 6.
- Test: `records.rs` `#[cfg(test)] mod tests`

**Interfaces**
- Produces:
  ```rust
  pub enum Source { PubChem { cid: u64 }, Lira { functional: String, quality: String, ionic: String, multiplicity: u8, max_force: f64, commit: String } }
  pub struct Record {
      pub name: String, pub synonyms: Vec<String>, pub source: Source,
      pub z: Vec<u8>, pub charge: Vec<i8>,            // per atom (formal)
      pub bonds: Vec<(usize, usize, u8)>,             // order 1–3
      pub xyz: Vec<[f64; 3]>,                         // Å
      pub xy: Vec<[f32; 2]>,                          // 2D layout, canvas units (a bond ≈ 1)
      pub stereo_defined: bool,
  }
  impl Record { pub fn formula_key(&self) -> (String, i32) /* (Hill formula, net charge) */; pub fn validate(&self) -> Result<(), String> }
  pub const FORMAT: &str = "ASMOLECULES 1";
  pub fn parse(text: &str) -> Result<Vec<Record>, String>;
  pub fn write(records: &[Record]) -> String;
  pub struct Library { /* records indexed by formula_key */ }
  impl Library { pub fn shipped() -> &'static Library; pub fn from_records(v: Vec<Record>) -> Library; pub fn candidates(&self, key: &(String, i32)) -> &[Record]; pub fn records(&self) -> &[Record] }
  pub fn parse_sdf(sdf: &str) -> Result<(Vec<u8>, Vec<i8>, Vec<(usize, usize, u8)>, Vec<[f64; 3]>), String>; // V2000, first molecule
  ```
- `validate` enforces spec §5:
  - Z ≤ 18 and ≤ 30 atoms;
  - each bond length is in [0.5, 3.5] Å;
  - bond indices are in range and not duplicated;
  - the `xyz` and `xy` lengths equal the atom count.

  Uniqueness of CIDs across the file is checked by Task 6's test.
- The text format beyond `FORMAT` is the implementer's choice. It must be line-oriented, human-diffable and round-trip exactly (`parse(write(v)) == v`).
- The 2D canvas scale: `xy` from a PubChem 2D SDF is rescaled so its mean bond length is 1.0 canvas unit. This rule is shared by Tasks 5, 6 and 7.

- [ ] **Step 1: Write the failing tests**

```rust
#[test] fn the_seed_file_parses_and_validates() {
    let lib = Library::shipped();
    assert!(lib.records().len() >= 6);
    for r in lib.records() { r.validate().unwrap_or_else(|e| panic!("{}: {e}", r.name)); }
}
#[test] fn the_format_round_trips() { let v = Library::shipped().records().to_vec(); assert_eq!(parse(&write(&v)).unwrap(), v); }
#[test] fn ammonium_and_hydroxide_carry_their_charges() {
    let nh4 = Library::shipped().records().iter().find(|r| r.name == "ammonium").unwrap();
    assert_eq!(nh4.formula_key(), ("H4N".into(), 1));
    let oh = Library::shipped().records().iter().find(|r| r.name == "hydroxide").unwrap();
    assert_eq!(oh.formula_key(), ("HO".into(), -1));
}
#[test] fn validate_refuses_out_of_range_entries() { /* Z 19, 31 atoms, a 4 Å bond, a bond to atom 99: each Err */ }
#[test] fn sdf_v2000_water_parses() { /* inline 3-atom V2000 block with M  CHG absent: z [8,1,1], bonds (0,1,1),(0,2,1), xyz as given */ }
#[test] fn sdf_charges_come_from_the_m_chg_line() { /* inline NH4+ block with "M  CHG  1   1   1": charge [1,0,0,0,0] */ }
```

- [ ] **Step 2: Run them and see them fail.** `cargo test -p app --release --bin app -- builder::records` fails because the module is missing.
- [ ] **Step 3: Implement `records.rs` and write the seed `molecules.txt`.** Get the seed's coordinates from PubChem's 3D and 2D SDFs for CIDs 962 (water), 297 (methane), 702 (ethanol), 241 (benzene), 223 (ammonium) and 961 (hydroxide).
  - **Fetching:** fetch them once by hand with `curl`, and record the URLs in the test module's doc.
  - **No PubChem 3D record:** if hydroxide has none, use `Source::Lira` placeholder values marked `commit = "seed"`, and note that Task 7 regenerates it.
  - **Embedding:** `Library::shipped()` parses `include_bytes!("../../data/molecules.txt")` once, via `std::sync::OnceLock`.
- [ ] **Step 4: Run the tests and see them pass.** Same command: all pass.
- [ ] **Step 5: Commit.** `feat(builder): molecule records, the shipped data file format and SDF parsing (builder database, Task 1)`

### Task 2: Fragments and exact atom mapping — **Opus**

**Files:**
- Create: `crates/app/src/builder/iso.rs`
- Test: `iso.rs` tests

**Interfaces**
- Consumes: `Drawing`, `Atom`, `Bond` (`builder/mod.rs`), and Task 1's `Record`.
- Produces:
  ```rust
  pub fn fragments(d: &Drawing) -> Vec<Vec<usize>>;                 // connected components, each sorted, in order of their lowest atom index
  pub fn drawing_key(d: &Drawing, frag: &[usize]) -> (String, i32); // same Hill formula + net charge rule as Record::formula_key
  pub const STEP_CAP: usize = 200_000;
  pub enum MapResult { Mapped(Vec<usize>) /* drawn frag position i → record atom */, NoMatch, CapHit }
  pub fn map_fragment(d: &Drawing, frag: &[usize], r: &Record) -> MapResult;
  ```

**Mapping rule (spec §6.2).** The mapping is a bijection that preserves:
- element and formal charge;
- adjacency;
- bond orders, with one exception: if orders differ, every differing bond must be in a ring, and each atom's sum of bond orders must agree. This is the Kekulé case.

The algorithm is VF2-style backtracking, with candidates ordered by (Z, degree, sum of bond orders). It counts steps and returns `CapHit` at `STEP_CAP`.

- [ ] **Step 1: Write the failing tests.** Use the seed records via `Library::shipped()`, and build the drawings with `Drawing::add_atom` / `bond`.
  - `ethanol_maps_in_any_atom_order`: three permutations of the drawn ethanol each return `Mapped(m)`. For every bond (a, b) in the fragment, `(m[a], m[b])` is a record bond, and Z is equal for every atom.
  - `benzene_matches_in_either_kekule_form`: two drawings with alternating doubles shifted by one both return `Mapped`.
  - `near_misses_do_not_match`: water missing one H, ethanol with O → S, and ethanol with an extra C–O ring bond each return `NoMatch`.
  - `charges_must_agree`: neutral NH₄ drawn (5 atoms, N charge 0) gives `NoMatch` against ammonium; with N charge +1 it gives `Mapped`.
  - `symmetric_molecules_match_inside_the_cap` (Review focus 2): methane and benzene each return `Mapped`, with a step count below `STEP_CAP / 100`. Expose `map_fragment_counting(..) -> (MapResult, usize)` as `pub(crate)` for this test.
  - `the_cap_returns_cap_hit`: with a test-only cap of 1 (`map_fragment_with_cap`), benzene gives `CapHit`.
  - `fragments_split_components`: two waters and a lone Na give 3 fragments in index order.
- [ ] **Step 2: Run them and see them fail.** `cargo test -p app --release --bin app -- builder::iso`
- [ ] **Step 3: Implement `iso.rs`.**
- [ ] **Step 4: Run them and see them pass.**
- [ ] **Step 5: Commit.** `feat(builder): fragments and exact atom mapping with the Kekulé rule (builder database, Task 2)`

### Task 3: The hand-off geometry and labels — **Opus**

**Files:**
- Create: `crates/app/src/builder/shape.rs`
- Modify: `crates/app/src/builder/ui.rs`:
  - `handoff_rows` (around :715–848): use the shape for `new` / `beside`, the fit check and the hand-off;
  - `footer_rows` (:662): the label line;
  - the rule-1 note: add "database shapes".
- Modify: `crates/app/src/builder/mod.rs`: the `Builder::embedded` cache becomes `shaped: Option<(Drawing, Shaped)>`.
- Test: `shape.rs` tests, plus `ui.rs` tests for the fit check.

**Interfaces**
- Consumes: Task 1's `Library` and `Record`; Task 2's `fragments`, `drawing_key`, `map_fragment` and `MapResult`; `embed::{embed, centre, place_beside, closest}`.
- Produces:
  ```rust
  pub enum FragmentSource { PubChem { name: String, cid: u64, stereo_note: bool }, Lira, Approximate(Why) }
  pub enum Why { NotInDatabase, Offline, LookingUp }
  pub struct Shaped { pub positions: Vec<[f64; 3]> /* Bohr, drawing order, centred as embed::centre does */, pub sources: Vec<(Vec<usize>, FragmentSource)> }
  pub trait RecordSource { fn find(&self, d: &Drawing, frag: &[usize]) -> Lookup; }
  pub enum Lookup { Found(Record, Vec<usize>), NotFound, Pending, Offline }
  pub fn shape(d: &Drawing, src: &dyn RecordSource) -> Shaped;
  pub fn footer_label(s: &Shaped) -> String;   // the Global Constraints strings, verbatim
  pub const FRAGMENT_GAP: f64 = 3.0;           // Bohr
  pub struct LocalOnly<'a>(pub &'a Library);   // a RecordSource: the shipped list only (Task 4 adds the cache and online)
  ```
- `shape` follows spec §6.1:
  1. `embed(d)` lays out the whole drawing.
  2. Each fragment with a `Found` record is replaced by its record's `xyz` (converted with `angstrom_to_bohr`) through the mapping, centred on that fragment's embedded centroid.
  3. Matched fragments are then translated apart until every pair of atoms in different fragments is ≥ `FRAGMENT_GAP`. Only matched fragments move. Pushing along the line between the fragment centroids is enough.
  4. `centre(&mut positions)` is applied last, as today.
  5. Lone atoms and fragments that are not found keep the `embed` positions.
- `Lookup::Pending` labels that fragment `Approximate(LookingUp)` and keeps the embedding. Task 4 disables Insert while any fragment is pending.

- [ ] **Step 1: Write the failing tests.**
  - `a_matched_hand_off_uses_the_record_shape`: for drawn ethanol, every pairwise distance in `shape(..).positions` equals the record's (Å → Bohr) to 1e-9 Bohr.
  - `an_unmatched_drawing_is_bit_identical_to_today`: for H₃⁺ (not in the seed), `positions` are bit-equal to `embed` + `centre`. Do this for one drawing and for a two-fragment unmatched drawing.
  - `two_waters_and_a_lone_atom_keep_their_gaps` (Review focus 3):
    - both waters map to the record (internal distances to 1e-9);
    - every pair of atoms in different fragments is ≥ 3.0 Bohr;
    - the lone atom's position equals the whole-drawing embedding's before the final centring, i.e. offset by the same centring shift.
  - `labels_read_as_the_spec_says`: one case per Global Constraints string: ethanol, an unmatched drawing, a `Pending` source (a test `RecordSource`), an `Offline` source, the mixed summary, and a stereo note.
  - `the_fit_check_uses_the_database_shape` (Review focus 4), in `ui.rs` tests: a test `RecordSource` returns ethanol stretched ×3. The reach used by the fit check equals `max_abs` of the shaped positions, not the embedding's.
- [ ] **Step 2: Run them and see them fail.** `cargo test -p app --release --bin app -- builder::`
- [ ] **Step 3: Implement `shape.rs` and wire `ui.rs` to `shape(&b.drawing, &LocalOnly(Library::shipped()))`.**
  - Recompute only when the drawing changes, as `embedded` did.
  - The footer shows `footer_label`, with per-fragment labels in the hover text when there are several fragments.
- [ ] **Step 4: Run them and see them pass.** All `builder::` tests pass, including the existing ones unchanged.
- [ ] **Step 5: Commit.** `feat(builder): hand off database shapes per fragment, label each source (builder database, Task 3)`

### Task 4: The online lookup, cache and setting — **Opus**

**Files:**
- Create: `crates/app/src/builder/online.rs`
- Modify:
  - `builder/mod.rs`: a `Builder::online: bool` field, default `true`, with the lookup state;
  - `ui/profile.rs`: save and load `builder_online = true|false`, display-only, not an `Action`;
  - `builder/ui.rs`: the palette checkbox and the Insert button state;
  - `main.rs`: a process-wide `online::set_network_allowed(false)` for `--replay` and `--screenshot`.
- Test: `online.rs` tests, plus `ui.rs` and profile tests.

**Interfaces**
- Consumes: Task 1's `Record`, `parse_sdf` and the 2D rescale rule; Task 2's `map_fragment`; Task 3's `RecordSource`, `Lookup` and `Shaped`.
- Produces:
  ```rust
  pub fn smiles(d: &Drawing, frag: &[usize]) -> String;  // Kekulé, explicit H as atoms ([H] where needed), charges as [N+], ring-closure digits
  pub trait Transport: Send + Sync { fn get(&self, url: &str, timeout: std::time::Duration) -> Result<String, NetError>; }
  pub enum NetError { Offline, Timeout, Status(u16), Malformed }
  pub struct UreqTransport;                 // ureq agent, rustls, user agent "atomic-sandbox-builder/<version>"
  pub fn set_network_allowed(on: bool);    // process-wide; false under --replay/--screenshot; cfg(test) default false
  pub fn network_allowed() -> bool;
  pub struct Cache { /* dir: user_root()/molecules */ }
  impl Cache { pub fn open(dir: PathBuf) -> Cache; pub fn find(&self, key: &(String, i32)) -> Vec<Record>; pub fn store(&self, r: &Record) -> std::io::Result<()> }
  pub struct Lookups { /* worker thread, requests keyed by (drawing generation, fragment) */ }
  impl Lookups { pub fn new(t: Box<dyn Transport>, cache: Cache) -> Lookups; pub fn request(&self, generation: u64, d: &Drawing, frag: &[usize]); pub fn poll(&self) -> Vec<(u64, Vec<usize>, Lookup)> }
  pub struct Layered<'a> { /* shipped → cache → Lookups results for this generation */ }  // impl RecordSource
  pub const DEBOUNCE: Duration = Duration::from_secs(1); pub const TIMEOUT: Duration = Duration::from_secs(5); pub const MIN_INTERVAL: Duration = Duration::from_millis(200);
  ```

**PubChem calls (PUG REST).** Use `MIN_INTERVAL` between requests.
1. `…/compound/smiles/<url-encoded smiles>/cids/TXT`
2. `…/compound/cid/<cid>/SDF?record_type=3d`
3. `…/compound/cid/<cid>/SDF?record_type=2d`
4. `…/compound/cid/<cid>/property/Title/TXT` for the name

**What counts as a find:**
- The first CID whose 3D SDF `map_fragment`s to the drawn fragment is the answer.
- A 404 on the 3D SDF counts as not in the database.
- `stereo_defined` is false when PubChem's SMILES had no stereo marks. A stereo-specific record sets `stereo_note`.

**Generations.** The drawing generation goes up on every drawing change, and results for an older generation are dropped.

- [ ] **Step 1: Write the failing tests.** Use a `Canned` transport, a URL → body map that counts calls, and a `FailOnCall` transport that panics if called.
  - `smiles_of_the_seed_molecules`: ethanol, NH₄⁺, benzene (Kekulé) and OH⁻ give strings that round-trip through a minimal test SMILES reader to the same graph. Keep the reader in the test module.
  - `a_drawing_found_online_maps_and_caches`: canned ethane responses (CID 6324, 3D and 2D SDF). `poll` returns `Found` mapped, the cache holds it afterwards, and a second lookup in a new `Lookups` with `FailOnCall` is answered from the cache.
  - `a_timeout_falls_back_offline`: a transport returning `Timeout` gives `Lookup::Offline`.
  - `a_late_answer_for_an_old_drawing_is_dropped` (Review focus 1): a request at generation 1, then the drawing changes (generation 2). A result tagged 1 never labels or places the generation-2 shape.
  - `a_corrupt_cache_file_is_refetched`: garbage in a cache file makes `find` skip it and a canned refetch succeed, with no panic.
  - `no_network_when_off_replay_or_test`: with the setting off, or `set_network_allowed(false)`, and with `FailOnCall`, lookups return `Offline` and never call. A `#[cfg(test)]` assert shows `network_allowed()` is false by default in tests.
  - `insert_waits_while_looking_up`, in `ui.rs` tests: with any fragment `Pending`, the Insert and Replace buttons are disabled and read `Insert (looking up…)`. After `TIMEOUT`, simulated with an injectable clock, they enable with the offline label.
  - `the_online_setting_round_trips_in_the_profile`.
- [ ] **Step 2: Run them and see them fail.** `cargo test -p app --release --bin app -- builder:: ui::tests::profile`
- [ ] **Step 3: Implement `online.rs` and wire it in.**
  - Debounce: a request is sent once the drawing has been unchanged for `DEBOUNCE`, and only for fragments the shipped list and cache don't know.
  - Checkbox label: `Look up molecules online (PubChem)`. Tooltip: `sends the molecule's name or structure string to NCBI's PubChem; off: no network`.
- [ ] **Step 4: Run them and see them pass.**
- [ ] **Step 5: Commit.** `feat(builder): PubChem lookup off the UI thread, the molecule cache and the online setting (builder database, Task 4)`

### Task 5: Search and stamping — **Opus**

**Files:**
- Create: `crates/app/src/builder/search.rs`
- Modify:
  - `builder/ui.rs`: the search box at the top of `palette_column` and its results list;
  - `builder/mod.rs`: `Builder::search_query: String`, plus a `stamp_record` method;
  - `screenshot.rs`: a new `--ui` key `builder=<query>`, which opens the builder, types the query and stamps the first result.
- Test: `search.rs` tests, plus `ui.rs` tests.

**Interfaces**
- Consumes: Task 1's `Library` and `Record`; Task 4's `Cache`, `Lookups`, `Transport` and `network_allowed()`.
- Produces:
  ```rust
  pub struct Hit { pub name: String, pub formula: String, pub source: &'static str /* "shipped" | "cached" | "PubChem" */, pub record: Option<Record> }
  pub const MAX_RESULTS: usize = 8;
  pub fn search_local(q: &str, lib: &Library, cache: &Cache) -> Vec<Hit>; // exact name, then prefix, then synonym, then formula; case-insensitive; Unicode subscripts folded to digits
  pub fn parse_query(q: &str) -> Query;   // enum Query { Empty, Cid(u64), Formula(String), Name(String) }
  impl Builder { pub fn stamp_record(&mut self, r: &Record); } // one checkpoint; atoms from r.xy placed right of the drawing's bounding box + 1.5 units; bonds/charges copied
  ```
- The online search is `…/compound/name/<url-encoded>/cids/TXT` or `…/compound/cid/<n>/…`, through Task 4's `Lookups`, with the same no-network rules. The "Search PubChem for '<q>'" row appears only when online is on and network is allowed.

- [ ] **Step 1: Write the failing tests.**
  - `exact_names_rank_first`: "ethanol" puts ethanol first; "eth" puts ethanol among the results; "C2H6O" finds ethanol by formula.
  - `odd_queries_never_panic_or_send_malformed_urls` (Review focus 5): queries "", "   ", "000702", "H₂O", "h2o", "ethanol; DROP", and a 300-character string.
    - `parse_query` never panics;
    - "000702" gives `Cid(702)`;
    - "H₂O" and "h2o" both find water;
    - every URL the online search would build (through a recording transport) is ASCII, percent-encoded and under the PubChem prefix.
  - `stamping_is_one_undo_step_beside_the_drawing`: stamp benzene next to a drawn water. There are 12 more atoms, one `undo()` restores the water exactly, and every stamped atom's x is greater than the water's max x.
  - `the_stamped_layout_has_unit_bonds`: the stamped benzene's mean drawn bond length is 1.0 ± 1e-6.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement it.**
  - The search box hint is `molecule name, formula or CID`.
  - Each result row shows the name, the formula and its source.
- [ ] **Step 4: Run them and see them pass.** Then render `--screenshot out.png --ui "builder=ethanol"` (CPU-only rendering is acceptable: `--backend cpu`) and look at the PNG. The search box, the stamped ethanol and the footer label must all be visible.
- [ ] **Step 5: Commit.** `feat(builder): search by name, formula or CID and stamp the molecule (builder database, Task 5)`

### Task 6: The shipped PubChem list — **Opus** (CPU and network, no GPU)

**Files:**
- Create: `crates/app/data/molecule-names.txt`, the reviewed name list (spec §5 categories, 1 000–2 000 names), one `name` or `name | CID` per line.
- Create: the generator `crates/app/src/builder/records.rs` `#[cfg(test)] mod generate`. It is `#[ignore = "dev tool: cargo test -p app --release --bin app -- builder::records::generate --ignored --nocapture"]`, uses `UreqTransport` and `set_network_allowed(true)`, and is the only test allowed network.
- Replace: `crates/app/data/molecules.txt`. Every `PubChem` row plus the seed's `Lira` rows, which are kept as they are until Task 7.
- Test: the `records.rs` integrity test.

**Interfaces**
- Consumes: Tasks 1 and 4.
- Produces: the full `Library::shipped()`.

**Generator rules**
1. For each name, resolve the CID and fetch the 3D and 2D SDFs at ≤ 5 requests per second.
2. Skip, and report, any entry that:
   - has no 3D record;
   - has Z > 18 or more than 30 atoms;
   - fails `validate`;
   - duplicates an earlier CID.
3. Each record's 2D layout must map to its 3D graph with `map_fragment`.
4. It prints counts kept and skipped, with reasons, and the file size.

- [ ] **Step 1: Write the failing test** `the_shipped_list_is_complete_and_sound`:
  - every record validates;
  - CIDs are unique;
  - ≥ 1 000 PubChem records;
  - every builder stamp from `chem::stamp`'s set and every app preset molecule within H–Ar that has a PubChem 3D shape is present;
  - each record's `xy` and `xyz` graphs agree;
  - the file is ≤ 3 MB, from `include_bytes!` length.
- [ ] **Step 2: Run it and see it fail.** The seed has 6 records.
- [ ] **Step 3: Write `molecule-names.txt` and run the generator.** The generator commits nothing itself; review its skip report.
- [ ] **Step 4: Run the integrity test and all `builder::` tests; they pass.** Note the file size and count in the report.
- [ ] **Step 5: Commit** the name list, the data file and the generator: `feat(builder): the shipped PubChem molecule list (<n> molecules, <size> MB) (builder database, Task 6)`

### Task 7: LIRA-relaxed entries — **Opus** (GPU: dispatch only when the coordinator says the GPU is free)

**Files:**
- Create: `crates/app/data/lira-molecules.txt`, the spec §5.1 graphs. One entry per line: name, synonyms, atoms with charges, bonds with orders, the multiplicity hint `auto`.
- Create: the generator in `crates/app/src/worker/tests.rs`, `#[ignore = "dev tool, GPU: cargo test -p app --release --bin app -- worker::tests::generate_lira_shapes --ignored --nocapture"]`.
- Modify: `crates/app/data/molecules.txt`. The `Lira` rows are regenerated and the seed's placeholder rows replaced.
- Test: in the `records.rs` integrity test, `lira_rows_are_relaxed_and_traceable`.

**Interfaces**
- Consumes:
  - `builder::embed::embed` and `builder::replace_scene`;
  - `SimWorker::spawn(scene, Backend::Gpu, gpu_ctx())`;
  - `Command::Dyn(DynCommand::Relax)` and the snapshot's `dynamics.relax` (`RelaxProgress { converged, verified_max_force, .. }`);
  - the pattern of `a_relax_runs_to_its_end_without_another_command` and `diag_pass_rate`, in `worker/tests.rs`.
- Produces: `Source::Lira { functional: "LSDA", quality: "Accurate", ionic: <the scene's ionic tag>, multiplicity, max_force, commit }` rows.

**The scene for each entry**
- the embedding of the drawn graph, through `replace_scene`;
- `grid_h = sigma = 0.15` (Accurate);
- `ionic = IonicPolicy::Accepted`;
- the entry's net charge;
- `multiplicity: None` (automatic);
- `grid_n = enlarge_n(max_abs, 0.15, 81)`.

**Running it**
- Relax until `converged == Some(true)`. A `Some(false)` or a 30-minute wall is reported and left out.
- `max_force` is `verified_max_force`, and the multiplicity is the one the solve settled on.
- If an entry has a PubChem 3D shape (look it up with Task 4's client), skip it and report it. PubChem wins, as spec §5.1 says.
- The 2D layout is the relaxed shape projected onto its two largest principal axes, then rescaled to a mean bond length of 1.0.

- [ ] **Step 1: Write the failing test** `lira_rows_are_relaxed_and_traceable`:
  - every `Lira` row has `max_force` ≤ the app's Relax tolerance (`sim::relax::FireParams::default()`'s force tolerance);
  - each row has a non-empty `commit` that is not `"seed"`, and `functional == "LSDA"`;
  - drawing LiH, BH₄⁻ and H₃⁺ maps to their rows;
  - `footer_label` reads `shape: relaxed with LIRA (LSDA, Accurate)`.
- [ ] **Step 2: Run it and see it fail.** The seed placeholders fail it.
- [ ] **Step 3: Write the §5.1 list and run the generator** (GPU, BelowNormal, alone). Paste its table into the task report:

  | entry | multiplicity | max_force | steps | wall | kept or skipped |
  |---|---|---|---|---|---|

- [ ] **Step 4: Run the integrity tests and `builder::`; they pass.**
- [ ] **Step 5: Commit.** `feat(builder): LIRA-relaxed shapes for molecules PubChem has no 3D shape for (builder database, Task 7)`

### Task 8: Docs, rule 1 and the full app suite — **Opus**

**Files:**
- Modify:
  - `CLAUDE.md`, rule 1's builder exception line (spec §3, verbatim addition): `and its molecule-shape database (PubChem records and LIRA-relaxed shapes)`;
  - `README.md`, the builder section: search, database shapes, the online setting, and the PubChem credit with a link to `https://pubchem.ncbi.nlm.nih.gov`;
  - `docs/dev-reference.md`: the builder bullet gets the new files, both generator commands, the cache path, the no-network rule, and the `--ui builder=<query>` key;
  - `DEVLOG.md`: the entry, at merge.

**Steps**

- [ ] **Step 1: Write the docs.**
- [ ] **Step 2: Run the gates, serially, and only when the coordinator says the GPU is free.**
  - `cargo test -p app --release`, the viewer / UI row, including the replays: all pass.
  - `cargo test -p app --release --bin app -- builder::`: all pass.
  - Before that, a CPU-only pre-check is allowed: `cargo test -p app --release --bin app -- builder:: ui::`.
- [ ] **Step 3: Render the checked screenshots and attach them to the report:**
  - `--ui "builder=ethanol"`;
  - `--ui "builder=lithium hydride"`.
- [ ] **Step 4: Commit.** `docs(builder): database shapes in the README, dev reference and rule 1 (builder database, Task 8)`

## Execution order

Tasks 1 → 2 → 3 → 4 → 5 → 6 run on the CPU now, with network for Task 6's generator. Task 7 waits for the GPU. Task 8's docs can land before Task 7, but its full suite waits for the GPU. The branch merges after Task 8.
