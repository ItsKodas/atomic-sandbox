# 2D molecule builder — design

**Date:** 2026-10-20 · **Status:** written for review · **Requested by:** Dakoda (user requests 2026-10-19 "a 2D blueprint molecule builder … it should assist you too using known lookup tables", 2026-10-20 "lets start working on it")

## 1. Intent

Getting a new molecule into the sandbox today means placing nuclei one by one with the Place tool (no sense of where they belong, slow to settle when played) or writing a `.scene` file by hand (what Claude has been doing for ethane, propane, aspirin…). The builder lets the user **sketch a molecule as a 2D structural formula** — atoms and bonds on a flat canvas, the way chemistry is drawn on paper — and turns it into a sensible **3D starting geometry** in the scene, which the real solver then relaxes. Success: drawing ethanol, acetic acid or benzene and getting a molecule that the bond detector names correctly after a Relax, in under a minute, without touching a file.

## 2. Goals, non-goals, anti-goals

**Goals**
- File ▸ **Builder…** opens a pop-up window (an `egui::Modal` like Open/Save, `ui/files.rs` is the model), its state kept while closed (reopening shows the last drawing).
- Draw: place atoms (H–Ar, `ptable::Z_ENABLED`), bond them, set bond orders 1/2/3, formal charges ±, delete, move, undo/redo.
- Assist (lookup tables, §3): fill hydrogens, valence warnings, group stamps.
- Live formula and name of what is drawn (the existing `names` module).
- 2D → 3D embedding (§5) and the hand-off to the scene: **Replace** the scene or **Insert** beside its atoms; optional **Relax after** (on by default).

**Non-goals (v1)**
- No 3D → 2D: the builder does not import the current scene for editing (a later "Edit in builder" is possible).
- No SMILES/MOL/SDF import or export.
- No stereo control (wedges/hashes), no cis/trans picking: the embedding's choice stands and the Relax decides.
- No elements beyond Z 18 (the solver's range).

**Anti-goals** (failure even if it "works")
- The tables leaking into physics: any value from §3 reaching `sim`, `qm`, `qm-gpu`, `providers`, forces, energies, the bond detector or the surrogate. They live in `crates/app/src/builder/` only and produce positions, nothing else.
- Implying the drawn bonds are the physics: the scene gets nuclei only; bonds shown afterwards are still the density's (M6). The builder says so.
- A geometry that breaks the run: overlapping atoms, atoms outside the cell or inside the walls — refused or fixed before hand-off (§6).

## 3. Rule 1 sign-off (CLAUDE.md "Per-element data allowed: Z and isotope mass only")

**Signed off by the user, 2026-10-20** ("Looks good" to the explicit question), under these conditions:
- Two per-element tables are allowed **in the builder only**: the usual **valences** and **covalent radii** (single-bond radii, Cordero et al. 2008, Dalton Trans. 2832; multiple-bond lengths from the single ones × 0.87 double / × 0.78 triple — the C–C/C=C/C≡C ratios 1.54/1.34/1.20 Å).
- They are a **drawing aid for the starting guess**: they set where nuclei are first put, and nothing else. They never enter forces, energies, the solve, the bond detector, the names' graph or any recorded physics.
- The solver corrects the guess (the Relax after hand-off, on by default).
- The builder's window states this in one line ("bond lengths and valences here only place the atoms; the solver decides the real geometry and bonds").

Valences (usual neutral valence; the warnings and fill use these, with formal charge shifting them by ±1 per the element's side): H 1, He 0, Li 1, Be 2, B 3, C 4, N 3, O 2, F 1, Ne 0, Na 1, Mg 2, Al 3, Si 4, P 3 (5 allowed), S 2 (4, 6 allowed), Cl 1, Ar 0. Charge rule: N, O, S, P with +1 gain a bond (NH₄⁺, H₃O⁺); C/N/O with −1 lose one (OH⁻); B with −1 gains one (BH₄⁻).

## 4. The window

Layout (left to right): a narrow **palette** column (element buttons H C N O F P S Cl and "more…" opening the periodic table popover already used by the toolbar, `ui/ptable.rs`; tools: Atom · Bond · Charge · Erase · Select/Move; stamps: CH₃ · OH · NH₂ · COOH · benzene), the **canvas** (most of the window), and a **footer**: live formula + name (`names`), the warnings count, the Rule-1 line, and the buttons **Fill hydrogens · Clear · Replace scene · Insert into scene**, with a "Relax after" checkbox.

Canvas interactions (the default tool is Atom with C selected):
- **Atom tool** (revised by the user's hand-test, 2026-10-20): **click an atom** to select it; with an atom selected, **click empty space** for a new atom of the selected element bonded to it — a ghost bond and atom follow the cursor — which becomes the selection, so clicks draw a chain. The new bond is snapped (user request 2026-10-20): one fixed drawn length (shorter to H) in the cursor's direction rounded to 30°, Alt-click places freely, and a snapped spot on an existing atom bonds to that atom (a ring closes). With nothing selected the click places a lone atom. **Click the selected atom again**: change it to the selected element. **Shift-click another atom**: bond it to the selection (a ring closure). **Drag an atom**: move it.
- **Right-click** deselects (right-drag pans); **Esc** deselects, and closes the window when nothing is selected.
- **Bond tool**: drag from atom to atom for a bond; click a bond to cycle 1 → 2 → 3 → 1 (the Atom tool cycles too).
- **Charge tool**: click an atom to cycle 0 → +1 → −1 → 0 (drawn as a superscript).
- **Erase tool / Delete**: removes an atom (and its bonds) or a bond.
- **Select/Move**: drag an atom; box-select and drag a group.
- **Wheel** zooms, **middle- or right-drag** pans. **Ctrl-Z / Ctrl-Y** undo/redo (a snapshot stack of the drawing).
- Implicit hydrogens are **not** drawn until Fill hydrogens adds them as real atoms (what you see is what goes into the scene). Carbon is drawn as a "C" label (no skeletal-formula shorthand in v1).
- An atom whose bond-order sum exceeds its allowed valence gets a red ring; one below it (after Fill) an amber ring, with the count in the footer. Hand-off is allowed with warnings (radicals are legitimate), with a confirm line naming them.

Stamps place a group bonded to the selected atom (or free if none): CH₃, OH, NH₂, COOH, and a benzene ring (alternating 1/2 bonds, regular hexagon).

## 5. 2D → 3D embedding

Input: atoms (Z, 2D canvas position, formal charge) and bonds (order). Output: 3D positions in Bohr. Pure function, unit-tested.
1. **Targets**: bond length r₀ = covalent-radius sum × the order factor (§3). A six-membered ring drawn as alternating single and double bonds (a Kekulé structure) is one aromatic ring: its six bonds get the mean of the single and double factors, (1 + 0.87)/2 (added in Part 3, so the §8 benzene test's equal C–C holds; C–C 1.42 Å, the Relax brings it to the solver's own). Ideal angle at each atom from its steric number SN = σ-bond count + lone pairs, lone pairs = (valence electrons − bond-order sum ∓ charge)/2 (main group): SN 2 → 180°, 3 → 120°, 4 → 109.47° (lone pairs compress to ≈ 104.5/107 as a guess). Double/triple bonds count one σ.
2. **Start**: the 2D layout scaled so the mean drawn bond = the mean r₀, z = small deterministic offsets (a hash of the atom index, ±0.3 Bohr) so the structure can leave the plane.
3. **Relax a simple geometric energy** (CPU, gradient descent with a step cap, ≤ 2000 iterations, converged on max force): bond springs to r₀; angle springs on the 1–3 distances to the ideal angle; a soft repulsion between non-bonded atoms closer than 1.2 × their r₀; for atoms with SN 3 (sp²), a planarity term keeping them in their neighbours' plane, and the same for both ends of a double bond (keeps ethylene and benzene flat). This is an editor's embedding, not physics (§3); the result is only a starting guess.
4. **Centre** at the origin (Replace) or **place beside** the existing atoms (Insert: translated along +x until the closest pair of atoms is ≥ 3 Bohr apart).

## 6. Hand-off to the scene

- **Replace**: a new `Scene` with the built nuclei (default isotopes), `charge` = Σ formal charges, `multiplicity` auto, the current grid, walls and ionic policy kept; sent as `Command::SetScene` through the `Action` layer (recorded like File ▸ Open; in a replay it is `# not replayable: builder`).
- **Insert**: one `AddNucleus` per atom (as Place does) plus `SetCharge` if the formal charges change the total.
- **Fit check** before either: every nucleus inside the walls and the cell margin (`tools::clamp_into_cell`'s limit). If it does not fit, the footer says so and offers **"enlarge the box"** — the smallest odd grid n at the current h that fits with 4 Bohr of margin (`Command::SetGrid`), with the memory guard's verdict (`providers::gpu_footprint_bytes`) shown before it is accepted.
- The grid/ionic-policy refusals that apply to any scene apply here too (e.g. O at a grid coarser than its h_req): the status chip says so as for File ▸ Open; the builder does not change the policy.
- **Relax after** (on by default): `Action::Relax` once the scene is in (paused). The window closes on Replace/Insert.

## 7. Code layout

`crates/app/src/builder/`: `mod.rs` (the drawing: atoms, bonds, charges, undo stack, operations), `chem.rs` (the two tables, valence/lone-pair arithmetic, fill hydrogens, warnings — the only home of the tables), `embed.rs` (§5), `ui.rs` (the window, palette, canvas painter, footer). `UiState` holds the builder's state (display state, not recorded); File menu entry, Ctrl-K entry "Molecule builder", README Controls row.

## 8. Testing (user preference 2026-10-20: offload hands-on testing to the user)

**Claude runs (CPU only, seconds):** unit tests of `chem` (fill H on C, N, O, charged N⁺/O⁻; warnings), `embed` (CH₄ angles ≈ 109.5° ± 3°, C₂H₄ planar within 0.05 Bohr and ≈ 120°, H₂O bent, benzene planar with equal C–C ± 2 %, no non-bonded pair closer than 1.5 Bohr, deterministic output), the hand-off's charge sum and fit check, and `names` on a built ethanol graph. `cargo test -p app --release --bin app -- builder::`. No GPU suites (nothing in sim/qm/providers changes).
**The user tries in the app:** the canvas feel (placing, chains by dragging, bond cycling, stamps, undo, zoom/pan), the window's look, Replace/Insert into real scenes, the Relax result and the name the bond detector gives afterwards.

## 9. Left to the builder (implementation calls)

Canvas drawing details (atom label sizes, bond line styles for double/triple, colours from `render::elements`), the exact energy weights and step sizes in §5 (tuned until the §8 tests pass), the palette's icon art, and the undo stack's depth (≥ 50).
