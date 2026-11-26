# Molecule builder: database shapes — design

**Date:** 2026-11-25 · **Status:** written for review · **Requested by:** the user (2026-11-25: "Ideally I want it to place everything into their bonds, that way when i insert into scene, i can instantly start training knowing everything is the correct distances and angles … We should use some already existing database/dictionary that has most molecules").

Builds on the 2D molecule builder (`2026-10-20-molecule-builder-design.md`, code in `crates/app/src/builder/`).

## 1. Intent

Today the builder turns a 2D drawing into 3D with a rough embedding: bond springs from covalent radii, VSEPR angles and a planarity term (`embed.rs`). The **Relax after** option (FIRE on the solver's own surface) is what makes the geometry right, and from a rough start that is a long run of full DFT steps before training can begin.

The user wants **Insert** and **Replace** to put a known molecule at its correct bond lengths and angles straight away, so training can start at once. They also want common molecules to be easier to get onto the canvas.

The answer is to look up what was drawn in a molecule database and use the database's 3D shape. Drawings the database doesn't know keep today's embedding, labelled as approximate.

**Success:**
- Draw ethanol, or type "ethanol", and Insert. The atoms land at PubChem's 3D shape, and the builder says so.
- Draw H₃⁺, and Insert gives today's embedding with "approximate shape: not in the database".

## 2. Decisions the user made (2026-11-25)

- **Accuracy:** a database shape 1–2 % off our LSDA minimum is fine ("close enough to begin training"). **Relax after** stays the way to settle onto our own surface. It stays an option, with its current default (on) unchanged.
- **Approach A:** look the drawing up by its structure (shipped list, then PubChem) and fall back to today's embedding. Not chosen:
  - a force-field generator (UFF) for every drawing;
  - both together.
  UFF can be added later as the fallback without changing A.
- **The builder stays 2D**, "to keep things simple and understandable". The correct 3D shape is produced only on Insert / Replace. There is no 3D preview in the builder.
- **Data source:** both. A list of common molecules ships with the app and works offline. A PubChem search covers everything else, and fetched records are kept for next time. This was the coordinator's proposal and the user didn't object.
- **Rule 1:** the user signed off ("yes sign it off") on extending the builder's starting-geometry exception (§3).
- **Molecules PubChem has no 3D shape for** use option 2, "our own relaxed shapes for a short list" (§5.1). The background:
  - PubChem3D's MMFF94s conformers exist only for H, C, N, O, F, Si, P, S, Cl, Br and I, and not for radicals.
  - So molecules with Li, Be, B, Na, Mg or Al, and radicals, have no PubChem shape. This was found by the builder check.
  - Not chosen: accepting the gap, and NIST CCCBDB experimental geometries (unclear terms for shipping).

## 3. Rule 1 (extends the 2026-10-20 sign-off)

Molecule records (atoms, bonds, charges, 2D and 3D coordinates) are allowed **in the builder only**, as a **starting-geometry aid**, like the valence and covalent-radius tables in `chem.rs`. The conditions:

- **They only place nuclei.** The records decide where the builder places nuclei on Insert / Replace, and what lands on the canvas from a search. Nothing else.
- **They never enter the physics.** No value from a record reaches `sim`, `qm`, `qm-gpu`, `providers`, `surrogate`, the forces, the energies, the bond detector, the names' graph or any recorded physics. The scene still receives nuclei (Z, position, default isotope) and the total charge, exactly as today's `Handoff`.
- **Where the code lives.** Only in `crates/app/src/builder/` and the shipped data file.
- **The LIRA-relaxed entries (§5.1)** are our own DFT results, not tabulated data, so rule 1 has nothing against them. They still follow the same "they only place nuclei" condition.
- **CLAUDE.md.** Rule 1's exception line for the builder gains "and its molecule-shape database (PubChem records and LIRA-relaxed shapes)", in the same commit as the code.

## 4. Goals, non-goals, anti-goals

**Goals**

1. **Shape lookup on hand-off.** When the drawing is a molecule the database knows, Insert / Replace place the drawn atoms at the record's 3D coordinates.
   - Each connected fragment of the drawing is looked up on its own (§6).
   - Fragments that are not found, and lone atoms, use today's `embed.rs`.
2. **Search by name.** A search box in the palette column takes a name, formula or PubChem CID and lists the matches.
   - Picking one puts it on the canvas as a tidy 2D drawing, using the record's own 2D layout, with all hydrogens explicit.
   - It is added beside whatever is already drawn, as one undo step.
3. **Shipped list.** Common molecules are available offline and answered instantly (§5).
4. **Online lookup.**
   - **PubChem** answers drawings and names not in the shipped list, over HTTPS with the `ureq` client the updater already uses.
   - **Cache.** Fetched records go to a local cache and are answered offline afterwards.
5. **Honest labels.** The footer says where the hand-off shape comes from, for each fragment. Examples:
   - `shape: PubChem "ethanol" (CID 702)`;
   - `shape: relaxed with LIRA (LSDA, Accurate)`, for the §5.1 entries;
   - `shape: approximate (not in the database) — Relax after recommended`;
   - `looking up online…`;
   - `shape: approximate (offline)`.

   The scene chip's label after a Replace is unchanged (the formula).

**Non-goals**

- 3D preview or 3D editing in the builder.
- Choosing a conformer, or stereo (wedges, cis/trans). The record's stereo and conformer stand, and the footer says when the drawing was stereo-ambiguous (§6.4).
- A force-field fallback (UFF): possible later, not now.
- Elements beyond the builder's range (Z ≤ 18, `ptable::Z_ENABLED`). Records with other elements are excluded from the list and refused from search.
- A "tidy layout" button for hand-drawn molecules. Hand drawings stay as drawn.
- SMILES / MOL / SDF import or export by the user. SMILES is used internally only (§7).

**Anti-goals** (failure even if it "works")

- **Database values leaking into physics** (§3).
- **A wrong shape presented as right.** Two cases:
  - an atom mapping that puts a drawn atom at another element's or another connectivity's position;
  - a fragment labelled "PubChem" that was really embedded.
  Mapping must be exact (§6.2). Anything uncertain falls back and says so.
- **Network surprises.**
  - Lookups must never block the UI thread or Insert indefinitely.
  - There must be no network traffic when the user hasn't drawn, searched or inserted, and none at all with online lookups turned off.
  - There must be no lookups during replays or tests.
- **A broken run.** Overlaps, atoms outside the cell, and the existing fit and enlarge checks must keep applying to the new shapes.

## 5. The shipped list

- **Source:** PubChem (NCBI). The compound records and the computed 2D and 3D conformers (PubChem3D, MMFF94s-optimised) are public-domain data. `docs/` and the README credit PubChem, with a link.
- **Content:**
  - roughly **1 000–2 000 common small molecules**, all within H–Ar;
  - each with **≤ 30 atoms including hydrogens**, and with a PubChem 3D conformer.
  Categories:
  - the builder's existing group stamps;
  - diatomics and common gases;
  - water, ammonia, methane and the other hydrides;
  - small organics up to C₆–C₈: alkanes, alkenes, alkynes, alcohols, aldehydes, ketones, acids, esters, ethers, amines, amides, nitriles;
  - aromatics: benzene, toluene, phenol, pyridine and others;
  - common solvents;
  - the amino acids (neutral form);
  - nucleobases, simple sugars, common ions that have a 3D conformer, and the molecules named in the app's presets and README.

  The builder picks the exact list. It is written as a plain name list in the repo, so it can be reviewed and grown.
- **How it's made:** a dev-only generator turns the name list into the data file, and the data file is committed. The generator is an `#[ignore]`d test or a small binary in the app crate's dev tooling, never shipped. It records each entry's name, CID, synonyms (a few), formula, charge, atoms, bonds with orders, and 2D and 3D coordinates in Å.
- **Size:** the data file is embedded in the app (`include_bytes!`). Budget ≤ 3 MB uncompressed; the builder reports the actual size.
- **Format:** the builder's call. It needs a version line. A test checks every entry: it parses, the element range holds, ≤ 30 atoms, every bond length is between 0.5 and 3.5 Å, there are no duplicate CIDs, and the 3D shape matches the 2D graph.

### 5.1 LIRA-relaxed entries (user's choice 2026-11-25, option 2)

**Scope**

A short list, roughly 30–50 entries, of molecules within H–Ar that PubChem has no 3D shape for. The categories:

- **Li, Be, B, Na, Mg and Al compounds:**
  - LiH, Li₂, LiF, LiCl, LiOH;
  - BeH₂, BeF₂, BeCl₂;
  - BH₃, BH₄⁻, B₂H₆, BF₃, BCl₃, B(OH)₃, NH₃BH₃;
  - NaH, Na₂, NaF, NaCl, NaOH;
  - MgH₂, MgF₂, MgCl₂, MgO;
  - AlH₃, AlH₄⁻, AlF₃, AlCl₃, Al₂Cl₆.
- **Small radicals and ions the app uses:** H₃⁺, OH, CH₃, NH₂, HO₂, NO, NO₂.
- **The rule:** any of these that does have a PubChem 3D shape takes PubChem's instead, and the generator reports which.
- **The list lives in the repo** beside the PubChem name list. Each entry is a small graph with atoms, bonds with orders, and formal charges.

**Generator**

A dev-only `#[ignore]`d harness. It does not ship and runs **on the GPU, so only when the user's GPU is free**. For each entry it:

1. builds today's `embed.rs` guess from the graph;
2. runs the app's own Relax (FIRE on the solver's surface) at the production defaults:
   - GPU backend;
   - **Accurate** (h = σ = 0.15 Bohr);
   - `Accepted` ionic policy;
   - the entry's charge;
   - automatic multiplicity: the lowest spin (electrons paired; one unpaired electron for an odd count), as `qm::System` chooses it — no multiplicity scan;
   - a box enlarged by the builder's own `enlarge_n` rule;
3. stops at Relax's own convergence criterion.

**Output**

Each entry records:

- the relaxed 3D coordinates;
- a 2D layout: the 3D shape projected onto its two largest principal axes and scaled to canvas units;
- the functional (LSDA), quality, ionic tag and multiplicity;
- the final max |F|;
- the commit it was generated at.

These rows go into the same shipped data file, marked with source `lira`. An entry whose Relax does not converge is left out and reported, never shipped half-relaxed.

**Later physics changes**

The shapes stay valid as starting guesses after later physics changes such as PBE or broken symmetry: like PubChem's, they only need to be close, and Relax settles the rest. Re-running the generator is optional, never required.

## 6. Matching a drawing to a record

### 6.1 Fragments

- The drawing is split into connected components.
- Each component with ≥ 2 atoms is matched on its own.
- A lone atom is placed as today.
- **Layout (coordinator's call, from the builder check).**
  1. `embed::embed` lays out the **whole drawing once**, exactly as today.
  2. Every unmatched atom keeps the position that layout gives it.
  3. Each matched fragment's atoms are replaced by its record shape, turned by the proper rotation (never a reflection, which would flip a stereo record's chirality) that best fits it, in the least-squares sense, to the embedding's positions for that fragment's atoms, then centred on the centroid the embedding gave that fragment. So each fragment keeps the orientation it was drawn in, and drawn arrangements between fragments (an H₂O···HF hydrogen bond) survive (user hand-test 2026-11-25; R16).
  4. Only matched fragments then move, pushed apart until every pair of atoms in different fragments is ≥ 3 Bohr, the same gap Insert already keeps from the scene's atoms.

  A drawing with no matched fragment therefore hands off bit for bit what it does today.

### 6.2 Exact atom mapping

A fragment matches a record only when there is a **graph isomorphism**: a one-to-one map of drawn atoms onto record atoms that keeps:

- each atom's element (Z);
- each atom's formal charge;
- connectivity: every drawn bond is a record bond and vice versa;
- bond orders, except that in rings, alternating single and double bonds may be placed differently (Kekulé forms). In practice the orders must match bond for bond, or both sides must have the same per-atom sum of bond orders with the difference confined to ring bonds.

Hydrogens are matched like any other atom. A drawing without hydrogens is a different molecule (a radical) and won't match a closed-shell record. That is correct and consistent with today's builder: drawn atoms are real atoms. The fallback label then suggests Fill hydrogens when the hydrogen-filled drawing would match.

- **Symmetric molecules:** any valid isomorphism gives the same shape up to relabelling.
- **Algorithm:** the builder's call. It must be exact. VF2-style backtracking is enough at ≤ 30 atoms, but needs a step cap with "not matched" as the outcome on hitting it.

**Placing the atoms.** Each drawn atom gets its mapped record atom's 3D position (Å → Bohr through the `units` crate). The record's orientation is kept as is. Insert's existing placement beside the scene's atoms then applies unchanged.

### 6.3 Lookup order

For each fragment:

1. **The shipped list.** Index it by formula and charge, then try isomorphism against each candidate.
2. **The cache**, the same way.
3. **PubChem, online only:**
   - write a SMILES for the fragment from the drawing (§7) and resolve it to CIDs;
   - fetch the CID's records (3D SDF and 2D SDF);
   - run the same exact mapping on the result.

   A record without a 3D conformer is "not in the database".
4. **The fallback:** today's `embed.rs` for that fragment, labelled approximate.

### 6.4 Stereo

The 2D drawing carries no stereo, so the SMILES has none either.

- If PubChem returns a record with undefined stereo and a 3D conformer, its conformer stands.
- If it returns only stereo-specific records, take the first CID that maps. The footer adds "stereo as in the database (CID n)".
- The shipped list stores one record per name, and its stereo stands.

## 7. Online lookup

- **Client:** `ureq` (already a dependency for the updater: rustls, no new runtime), over HTTPS to PubChem PUG REST (`pubchem.ncbi.nlm.nih.gov/rest/pug/…`). Name, CID and SMILES lookups use the documented endpoints. The builder picks the exact calls and keeps to PubChem's published rate limit (≤ 5 requests per second).
- **Off the UI thread.** A worker thread handles lookups. The builder window shows `looking up online…` and stays fully usable.
  - A drawing change cancels the pending result: it is ignored on arrival.
  - Timeout: **5 s** per lookup. On timeout or a network error, the fragment falls back with `approximate (offline)`.
- **When it runs:**
  - **Search:** only on the user's search. The "Search PubChem" row is offered unless the shipped list or cache has an exact name or CID match (Task 5 review, coordinator ruling R12). The combined list stays at 8 or fewer, and PubChem rows replace the lowest-ranked local rows.
  - **Drawn fragments:** on a drawn fragment that the shipped list and cache don't know, once the drawing has been unchanged for **1 s**, so Insert usually finds the answer ready.
  - **Insert during a lookup:** the button reads "Insert (looking up…)" and is disabled until the lookup ends or times out (≤ 5 s).
- **Setting:** "Look up molecules online (PubChem)" in the builder's palette column, default **on**. Its tooltip says what is sent: the molecule's name or structure string, to NCBI's PubChem. Off means no network traffic at all. The setting is display-only (`UiState`, persisted like other display settings), not an `Action`.
- **Cache:**
  - **Where:** fetched records live in the per-user root, `%LOCALAPPDATA%\atomic-sandbox\molecules\`, in every data mode, like `cache\pseudo`.
  - **Format:** the same record format as the shipped list, one file per CID, plus a small index keyed by formula and charge.
  - **Corrupt files:** a corrupt cache file is skipped and refetched, never a crash.
- **Replays and tests:** no network at all.
  - Builder hand-offs stay unreplayable, as today: `replay::unreplayable_kind` returns "builder" and a test pins it. A replay never reaches the builder.
  - Under `--replay`, `--screenshot` and in tests, online lookups are off, whatever the setting says.
  - Tests that cover the online path use canned PubChem responses.

## 8. The window (changes only)

- **Palette column:** a search box at the top (hint "molecule name, formula or CID").
  - **Results:** up to 8, shipped and cached first, then a "Search PubChem for '…'" row when online is on.
  - **Each row:** name, formula and source.
  - **Picking a row:** stamps the molecule onto the canvas, as in Goal 2.
  - **The online setting** goes below.
- **Footer:** beside the live formula and name, one shape line per hand-off.
  - One fragment: the single label from §4 Goal 5.
  - Several fragments: a compact summary, "shape: 2 from PubChem, 1 approximate", with each fragment's label in the tooltip.
- **Fit check:** it uses the shape the hand-off will use, the database's once known. Today's enlarge-the-box offer is unchanged.
- **Unchanged:** the one-line rule-1 note ("bond lengths and valences here only place the atoms…") stays, with "database shapes" added.

## 9. Tests

- **Mapping:**
  - ethanol drawn in three atom orders maps exactly;
  - benzene drawn in either Kekulé form matches;
  - a drawing missing one H, a swapped element and an extra bond each do not match;
  - a charged drawing (NH₄⁺, OH⁻) matches only the charged record;
  - the step cap gives "not matched".
- **Placement:**
  - after a matched hand-off, the drawn atoms' positions equal the record's (Å → Bohr) to 1e-9 Bohr;
  - in a two-fragment drawing, each fragment matches and the fragments end ≥ 3 Bohr apart;
  - a drawing with no matched fragment hands off bit-identically to today;
  - in a mixed drawing, the unmatched atoms keep their whole-drawing embedding positions.
- **Data file:**
  - the integrity test from §5;
  - a size check against the budget;
  - for `lira` rows, the recorded max |F| is below Relax's tolerance, and the provenance fields are present;
  - drawing LiH, BH₄⁻ and H₃⁺ matches their `lira` rows and shows the LIRA label.
- **Online, with canned responses:**
  - a SMILES lookup is resolved and mapped;
  - a timeout falls back with `approximate (offline)`;
  - a drawing change while pending discards the result;
  - the cache is written and then answered offline;
  - a corrupt cache file is refetched.
- **No-network guarantees:** with the setting off, under `--replay` and in tests, the lookup layer makes no request. Enforce this with a test transport that fails on any call.
- **UI:**
  - the footer labels for each case;
  - the Insert button state during a lookup;
  - search results ordering;
  - picking a result stamps it as one undo step.
- **Screenshot:** a `--screenshot --ui` key for the builder with a search result and a footer label, rendered and checked.
- **Gate:** `cargo test -p app --release` (viewer / UI row) and `cargo test -p app --release --bin app -- builder::`. No solver, surrogate or dynamics suites: nothing outside `app` changes.

## 10. Docs

- **README:** the builder section gets search, database shapes, the online setting and the PubChem credit.
- **`docs/dev-reference.md`:** the data file, the generator command, the cache location, and the no-network rule for replays and tests.
- **CLAUDE.md:** rule 1's exception line (§3).
- **DEVLOG:** at merge.

## 11. Left to the builder

- The data file's format and the exact shipped name list, within §5's budget.
- The isomorphism algorithm and its step cap.
- The SMILES writer (Kekulé form, explicit hydrogens as bracket atoms or counts, charges). It must round-trip the test molecules through PubChem's canned responses.
- How fragments are pushed apart (§6.1), within its 3 Bohr rule.
- The search ranking within §8 (exact name first, then prefix, then synonyms).
- The exact §5.1 list, within its categories. Common names and synonyms for `lira` entries (e.g. "borane", "diborane", "lithium hydride") so that search finds them.
