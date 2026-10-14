# UI redesign: Appendix A checklist (2026-10-14)

The acceptance checklist of the UI redesign: every row of Appendix A of the mockup
(`docs/superpowers/notes/2026-10-14-ui-mockup.html`, 148 controls and readouts of the old side
panel, the atom card, the overlays and the mouse/keyboard input) with where it lives now. File
references are relative to `crates/app/src/` and point at the line that draws (or records) the
item; this table is generated from a list of literal anchors, so a row only reads `done` when its
anchor text exists at that line. Tooltip texts of the old panel are kept verbatim on the moved
controls.

**184 / 184 rows done** (148 from Appendix A, then the M5b viewer extras merged from `main` as rows 149–164, the drag pull — user request 2026-10-14, branch `drag-force-wheel` — as row 165, the M6 branch's edits to the old `ui.rs`, ported when `main` was merged into `m6-tools`, as rows 166–168, and the M6 Task 10 controls — Appendix B's Q6–Q8 and the periodic table — as rows 169–179, and the molecule name style — user request 2026-10-15 — as row 180, and the sampling-plan presets — user request 2026-10-15, branch `m6-sampling-presets` — as row 181). Rows 1–168's line numbers are as of the merge (7d8d6af); Task 10's and Task 8's edits shifted some of them (the anchors named in each row still identify the place); rows 169–180 are as of Task 10's review fix round 1. The second merge of `main` into `m6-tools` (2026-10-14) brought main's two new rows, which main had numbered in place as **75a** (temperature scale K / °C / °F) and **81a** (Scene ▸ Quality): they are kept under those numbers (renumbering them after 180 would separate them from the controls they extend; no number overlaps 166–180), and main's edits to rows 80–81 (the `empty` preset), 149–151 (walls on by default) and 164 (`--no-walls`) replace the old text of those rows; their line numbers are as of that merge.

Abbreviations as in the mockup: TB = bottom toolbar, Dock ▸ X = right dock tab, Menu = top menu,
Overlay = drawn in the 3D view.

| # | Old item | New place | Status | Where (file:line) |
|---|---|---|---|---|
| 1 | heading "Atomic Sandbox" | Top bar brand + scene chip | done | `ui/menus.rs:93`, `ui/menus.rs:104` |
| 2 | status line + spinner | Overlay status chip (all states kept) | done | `ui/overlays.rs:92`, `ui/overlays.rs:138` |
| 3 | UI error (red) | Status chip red + Log (error) | done | `ui/overlays.rs:152`, `ui/logbook.rs:169` |
| 4 | "paused by a solver failure" | Status chip red + Log + strip count | done | `ui/overlays.rs:156`, `ui/logbook.rs:180` |
| 5 | "refused: …" (amber) | Status chip (amber line) and Log tab "last refusal" line, in place while the driver holds it + toast above the toolbar (re-raised per attempt) + Log entry | done | `ui/overlays.rs:121`, `ui/dock/log.rs:10`, `ui/logbook.rs:187`, `ui/overlays.rs:235` |
| 6 | neutral status (saved / loaded / switched; "paused: … press Play to continue") | Status chip line until replaced or cleared (amber when it asks for something) + toast per set_status + Log (status) | done | `ui/overlays.rs:121`, `ui/mod.rs:307`, `ui/logbook.rs:166`, `ui/overlays.rs:235` |
| 7 | solver radio · CPU | TB Solver segmented | done | `ui/toolbar.rs:629` |
| 8 | solver radio · GPU | TB Solver segmented, "no GPU adapter" | done | `ui/toolbar.rs:634` |
| 9 | solver radio · estimated | TB Solver segmented, tooltip names the teacher | done | `ui/toolbar.rs:37` |
| 10 | honesty line | Dock ▸ Model first line + i; Help ▸ Physics, honestly | done | `ui/dock/model.rs:18`, `ui/menus.rs:643` |
| 11 | this step (surrogate / fallback / DFT) | TB Solver sub-line + status chip pill + Dock ▸ Model | done | `ui/dock/model.rs:269`, `ui/toolbar.rs:652`, `ui/overlays.rs:180` |
| 12 | uncertainty vs bound | Dock ▸ Model (bar + numbers, tooltip kept) | done | `ui/dock/model.rs:272`, `ui/dock/model.rs:366` |
| 13 | trusted (five variants) | TB trust light + Dock ▸ Model + strip dot | done | `ui/mod.rs:622`, `ui/dock/model.rs:289`, `ui/dock/mod.rs:59` |
| 14 | samples (+ refit pending) | Dock ▸ Model | done | `ui/dock/model.rs:296` |
| 15 | model epoch | Dock ▸ Model + TB sub-line | done | `ui/dock/model.rs:299`, `ui/toolbar.rs:686` |
| 16 | last fit | Dock ▸ Model | done | `ui/dock/model.rs:302` |
| 17 | last refit | Dock ▸ Model | done | `ui/dock/model.rs:309` |
| 18 | fallbacks % of last 100 | Dock ▸ Model (bar) + TB sub-line | done | `ui/dock/model.rs:316`, `ui/toolbar.rs:686` |
| 19 | teacher gap | Dock ▸ Model | done | `ui/dock/model.rs:327` |
| 20 | last audit (verdict, gap, ms, lag, in flight) | Dock ▸ Model; audit lag also in Readouts | done | `ui/dock/model.rs:335`, `ui/dock/readouts.rs:176` |
| 21 | mismatch work | Dock ▸ Model | done | `ui/dock/model.rs:357` |
| 22 | provenance mismatch warning | Dock ▸ Model warning slot + Log | done | `ui/dock/model.rs:36`, `ui/logbook.rs:244` |
| 23 | autoload refused warning | Dock ▸ Model warning slot + Log | done | `ui/dock/model.rs:39`, `ui/logbook.rs:251` |
| 24 | "Forget it and sample this scene" | Dock ▸ Model, in the warning | done | `ui/dock/model.rs:40` |
| 25 | sampling progress bar | Sampling card + status chip + Dock ▸ Model | done | `ui/dock/model.rs:131`, `ui/overlays.rs:208` |
| 26 | last teacher energy · skipped · failed | Sampling card + Dock ▸ Model | done | `ui/dock/model.rs:137` |
| 27 | last pass summary | Dock ▸ Model + Log (pass) | done | `ui/dock/model.rs:53`, `ui/logbook.rs:226` |
| 28 | long first-pass offer (amber) | Offer card in the view + Dock ▸ Model | done | `ui/dock/model.rs:143`, `ui/overlays.rs:27` |
| 29 | offer "Start anyway (≈ t)" | Offer card (and Model tab) | done | `ui/dock/model.rs:156` |
| 30 | DFT audit every n | Dock ▸ Model ▸ Audits | done | `ui/dock/model.rs:69` |
| 31 | "(solver: n)" | Dock ▸ Model, beside the field | done | `ui/dock/model.rs:81` |
| 32 | audits-off unsafe warning | Dock ▸ Model, under the field | done | `ui/dock/model.rs:85` |
| 33 | plan · perturbations | Dock ▸ Model ▸ Sampling plan | done | `ui/dock/model.rs:232` |
| 34 | plan · σ_r | Dock ▸ Model ▸ Sampling plan | done | `ui/dock/model.rs:236` |
| 35 | plan · hot burst steps | Dock ▸ Model ▸ Sampling plan | done | `ui/dock/model.rs:242` |
| 36 | plan · hot burst T | Dock ▸ Model ▸ Sampling plan | done | `ui/dock/model.rs:243` |
| 37 | plan · scan points / pair | Dock ▸ Model ▸ Sampling plan | done | `ui/dock/model.rs:248` |
| 38 | plan · seed | Dock ▸ Model ▸ Sampling plan | done | `ui/dock/model.rs:254` |
| 39 | Start sampling / Sample more / armed "Start anyway" | Dock ▸ Model primary button (two clicks above 2 min) | done | `ui/dock/model.rs:161`, `ui/actions.rs:287` |
| 40 | plan cost "n solves, ≈ t" | Dock ▸ Model, beside the button | done | `ui/actions.rs:305` |
| 41 | Cancel | Sampling card + Dock ▸ Model | done | `ui/dock/model.rs:181`, `ui/overlays.rs:222` |
| 42 | Forget model (two clicks within 4 s) | File ▸ Estimated model + Dock ▸ Model | done | `ui/dock/model.rs:216`, `ui/menus.rs:223` |
| 43 | Save model | File ▸ Estimated model + Dock ▸ Model | done | `ui/dock/model.rs:197` |
| 44 | Load model | File ▸ Estimated model + Dock ▸ Model | done | `ui/dock/model.rs:204` |
| 45 | model path label | File ▸ Estimated model header + Dock ▸ Model | done | `ui/menus.rs:220`, `ui/dock/model.rs:101` |
| 46 | Play / Pause | TB ▸ Time primary button (Space; sampling reason kept) | done | `ui/toolbar.rs:480` |
| 47 | Step | TB ▸ Time | done | `ui/toolbar.rs:499` |
| 48 | Rewind | TB ▸ Time | done | `ui/toolbar.rs:481` |
| 49 | speed slider (steps/tick, log) | TB ▸ Speed (slider + typed field) | done | `ui/toolbar.rs:312` |
| 50 | "watchable (≈ fs/s)" | TB ▸ Speed sub-line | done | `ui/toolbar.rs:330` |
| 51 | measured fs/s (last 2 s) | TB ▸ Speed (large) + Readouts tile and Time and cost | done | `ui/toolbar.rs:322`, `ui/dock/readouts.rs:76` |
| 52 | fs/s this tick | TB ▸ Speed sub-line + Readouts | done | `ui/toolbar.rs:336`, `ui/dock/readouts.rs:151` |
| 53 | steps last tick | Readouts ▸ Time and cost | done | `ui/dock/readouts.rs:153` |
| 54 | ms / step | Readouts ▸ Time and cost | done | `ui/dock/readouts.rs:155` |
| 55 | dt | TB ▸ dt + Physics menu | done | `ui/menus.rs:440`, `ui/toolbar.rs:345` |
| 56 | t = fs, step N | Status chip + Readouts | done | `ui/overlays.rs:188`, `ui/dock/readouts.rs:146` |
| 57 | dt warning (amber) | TB dt field amber + Readouts ▸ Health + Log | done | `ui/menus.rs:442`, `ui/dock/readouts.rs:195` |
| 58 | T | Readouts tile + status chip | done | `ui/dock/readouts.rs:47`, `ui/overlays.rs:191` |
| 59 | KE (nuclei) | Readouts ▸ Energies (Ha / eV / both) | done | `ui/dock/readouts.rs:94` |
| 60 | E_KS | Readouts ▸ Energies | done | `ui/dock/readouts.rs:95` |
| 61 | E total | Readouts ▸ Energies (bold) | done | `ui/dock/readouts.rs:96` |
| 62 | E conserved | Readouts ▸ Energies | done | `ui/dock/readouts.rs:98` |
| 63 | work done by tools | Readouts ▸ Energies | done | `ui/dock/readouts.rs:105` |
| 64 | model refit (estimated) | Readouts ▸ Energies, indented | done | `ui/dock/readouts.rs:124` |
| 65 | model force mismatch (estimated) | Readouts ▸ Energies, indented | done | `ui/dock/readouts.rs:130` |
| 66 | drift (coloured) | Readouts tile + strip dot (1e-4 / 1e-3) | done | `ui/dock/readouts.rs:64`, `ui/dock/mod.rs:50` |
| 67 | per step (model ms / SCF steps, readbacks, ms) | Readouts ▸ Time and cost | done | `ui/dock/readouts.rs:161` |
| 68 | sparkline (E total by source, T, fallback dots) | Dock ▸ Graph (large) + mini in Readouts | done | `ui/dock/graph.rs:45`, `ui/dock/readouts.rs:188` |
| 69 | sparkline ranges + "last N fs" | Dock ▸ Graph | done | `ui/dock/graph.rs:129` |
| 70 | source legend | Dock ▸ Graph | done | `ui/dock/graph.rs:135` |
| 71 | box warning (amber / red) | Readouts ▸ Health + Log + strip dot | done | `ui/dock/readouts.rs:191`, `ui/logbook.rs:192` |
| 72 | dynamics notes (last 3) | Readouts ▸ Health (the 3 newest, as before) + Log (all) | done | `ui/dock/readouts.rs:202`, `ui/logbook.rs:99` |
| 73 | thermostat on/off | Physics menu + status chip "CSVR T K" | done | `ui/menus.rs:467`, `ui/overlays.rs:193` |
| 74 | thermostat target T | Physics menu | done | `ui/menus.rs:472` |
| 75 | thermostat τ | Physics menu | done | `ui/menus.rs:476` |
| 75a | temperature scale K / °C / °F (display only; user request 2026-10-14): T readout, status chip, sparkline range, thermostat target, hot-burst T; typed values clamped at absolute zero | Physics ▸ Thermostat (segmented, like Ha/eV/both); Ctrl K "celsius" / "fahrenheit" | done | `ui/menus.rs:584`, `ui/menus.rs:597`, `ui/mod.rs` `TempUnits`, `units` conversions; tests `temperatures_show_in_the_chosen_scale_and_stay_kelvin_inside`, `the_thermostat_target_is_typed_in_the_chosen_scale` |
| 76 | gravity "g = 9.8 m/s² (−y)" | Physics menu | done | `ui/menus.rs:494` |
| 77 | gravity "off" | Physics menu | done | `ui/menus.rs:491` |
| 78 | COM correction | Physics menu | done | `ui/menus.rs:520` |
| 79 | field magnitude readout | Physics menu, under the field | done | `ui/menus.rs:510` |
| 80 | preset combo (empty, h, h2, he, lih, h2o — `empty` added 2026-10-14) | File ▸ Presets + scene chip | done | `ui/menus.rs:177`, `ui/menus.rs:138`; test `the_presets_popup_lists_empty_first_and_sends_its_scene` |
| 81 | Reset | File ▸ Presets + scene chip | done | `ui/menus.rs:187` |
| 81a | Quality: Fast / Balanced / Accurate (default) / Precise + Custom h, with grid, ≈ cost and H₂ accuracy per level (user request 2026-10-14) | Scene ▸ Quality; Ctrl K "Quality" | done | `ui/menus.rs` `quality_controls`, `quality.rs`; tests `clicking_a_quality_level_sends_the_grid_edit`, `quality_changes_keep_the_box_and_leave_sigma_alone` |
| 82 | tool Orbit | TB ▸ Tool | done | `ui/toolbar.rs:517` |
| 83 | tool Place | TB ▸ Tool | done | `ui/toolbar.rs:518` |
| 84 | tool Select | TB ▸ Tool | done | `ui/toolbar.rs:519` |
| 85 | element to place (H–Ne) | TB ▸ element chips (pick → Place); compact flyout | done | `ui/toolbar.rs:535`, `ui/toolbar.rs:562`, `ui/actions.rs:102` |
| 86 | "no selection" hint | TB ▸ Selection placeholder + Dock ▸ Selection | done | `ui/toolbar.rs:386`, `ui/dock/selection.rs:12` |
| 87 | "selected: X #i at (x, y, z) Bohr" | TB ▸ Selection label + Dock ▸ Selection | done | `ui/toolbar.rs:376`, `ui/dock/selection.rs:24` |
| 88 | Delete selected | TB ▸ Selection + Dock ▸ Selection | done | `ui/toolbar.rs:616`, `ui/dock/selection.rs:47` |
| 89 | Focus selected | TB ▸ Selection + Dock ▸ Selection | done | `ui/toolbar.rs:608`, `ui/dock/selection.rs:44` |
| 90 | pinned | TB ▸ Selection Pin + Dock ▸ Selection | done | `ui/toolbar.rs:600`, `ui/dock/selection.rs:38` |
| 91 | hint "… edits pause the run and restart its time" | Scene menu, top line | done | `ui/menus.rs:234` |
| 92 | charge | Scene ▸ Electrons + scene chip | done | `ui/menus.rs:238` |
| 93 | multiplicity auto | Scene ▸ Electrons | done | `ui/menus.rs:249` |
| 94 | multiplicity value | Scene ▸ Electrons | done | `ui/menus.rs:250` |
| 95 | σ auto = h | Scene ▸ Electrons | done | `ui/menus.rs:260` |
| 96 | σ value | Scene ▸ Electrons | done | `ui/menus.rs:262` |
| 97 | grid n | Scene ▸ Grid (odd n only) | done | `ui/menus.rs:280` |
| 98 | grid h | Scene ▸ Grid | done | `ui/menus.rs:287` |
| 99 | cost readout | Scene ▸ Grid | done | `ui/menus.rs:296` |
| 100 | Apply grid | Scene ▸ Grid (enabled with a pending edit) | done | `ui/menus.rs:289` |
| 101 | density volume / isosurface | View ▸ Density + quick switch in the view | done | `ui/menus.rs:404`, `ui/overlays.rs:272` |
| 102 | gamma | View ▸ Density (disabled for isosurface) | done | `ui/menus.rs:410` |
| 103 | opacity | View ▸ Density (disabled for isosurface) | done | `ui/menus.rs:411` |
| 104 | iso (× ρ max) | View ▸ Density (disabled for volume) | done | `ui/menus.rs:412` |
| 105 | cell | View ▸ Display aids | done | `ui/menus.rs:421` |
| 106 | floor grid | View ▸ Display aids | done | `ui/menus.rs:422` |
| 107 | axes | View ▸ Display aids | done | `ui/menus.rs:423` |
| 108 | smooth motion | View ▸ Display aids | done | `ui/menus.rs:431` |
| 109 | placement plane | View ▸ Display aids | done | `ui/menus.rs:426` |
| 110 | spacing 1 Bohr / 1 Å | View ▸ Display aids | done | `ui/menus.rs:429` |
| 111 | E_KS (breakdown) | Readouts ▸ Energy breakdown and solver (collapsible) | done | `ui/dock/readouts.rs:236` |
| 112 | kinetic (electrons) | same collapsible | done | `ui/dock/readouts.rs:237` |
| 113 | external | same collapsible | done | `ui/dock/readouts.rs:238` |
| 114 | Hartree | same collapsible | done | `ui/dock/readouts.rs:239` |
| 115 | xc | same collapsible | done | `ui/dock/readouts.rs:240` |
| 116 | nuclear | same collapsible | done | `ui/dock/readouts.rs:241` |
| 117 | note "smeared nuclei (σ = …)" | same collapsible | done | `ui/dock/readouts.rs:244` |
| 118 | surrogate / fallback breakdown note | same collapsible | done | `ui/dock/readouts.rs:246` |
| 119 | GPU f32 scatter note | same collapsible | done | `ui/dock/readouts.rs:251` |
| 120 | SCF steps | same collapsible | done | `ui/dock/readouts.rs:259` |
| 121 | eigen residual (final) | same collapsible | done | `ui/dock/readouts.rs:262` |
| 122 | Poisson residual | same collapsible | done | `ui/dock/readouts.rs:263` |
| 123 | solve (ms, readbacks, engine) | same collapsible | done | `ui/dock/readouts.rs:267` |
| 124 | grid | same collapsible | done | `ui/dock/readouts.rs:271` |
| 125 | ρ max | same collapsible | done | `ui/dock/readouts.rs:275` |
| 126 | evaluation notes | same collapsible + Log | done | `ui/dock/readouts.rs:278`, `ui/logbook.rs:219` |
| 127 | evaluation warnings (amber) | Readouts ▸ Health + Log | done | `ui/dock/readouts.rs:199`, `ui/logbook.rs:219` |
| 128 | scene path field | File ▸ Scene file | done | `ui/menus.rs:197` |
| 129 | Save | File ▸ Scene file | done | `ui/menus.rs:199` |
| 130 | Load (+ .model sidecar) | File ▸ Scene file (sidecar rule and "sampling was running" kept) | done | `ui/menus.rs:206`, `ui/actions.rs:174` |
| 131 | atom card title | Overlay card top-right of the view + Dock ▸ Selection; card toggle (new) | done | `atom_card.rs:42`, `ui/dock/selection.rs:85` |
| 132 | "nucleus #i: p protons · n neutrons" | Card + Dock ▸ Selection | done | `atom_card.rs:90`, `ui/dock/selection.rs:27` |
| 133 | "z electrons (neutral atom), shells …" | Card + Dock ▸ Selection | done | `atom_card.rs:58`, `ui/dock/selection.rs:30` |
| 134 | Bohr schematic drawing | Card | done | `atom_card.rs:93` |
| 135 | "schematic — not to scale" | Card | done | `atom_card.rs:126` |
| 136 | shared-electrons caveat | Card (line + i) + Selection tab + Help | done | `atom_card.rs:67`, `ui/dock/selection.rs:86` |
| 137 | cell box (amber wire) | Overlay, unchanged (toggle #105) | done | `overlay.rs:41` |
| 138 | floor grid | Overlay, unchanged (toggle #106) | done | `overlay.rs:48` |
| 139 | axes | Overlay + corner gizmo (new) (toggle #107) | done | `overlay.rs:53`, `ui/overlays.rs:297` |
| 140 | placement plane (blue) | Overlay in Place mode (toggle #109) | done | `overlay.rs:56` |
| 141 | distance guides | Overlay (through the frame's viewport, clipped to it) + list in Dock ▸ Selection | done | `viewer.rs:923`, `ui/dock/selection.rs:58` |
| 142 | placement marker cross | Overlay | done | `viewer.rs:929` |
| 143 | drag spring ring + tether | Overlay | done | `viewer.rs:884` |
| 144 | pin square | Overlay | done | `viewer.rs:872` |
| 145 | faded-density label | Overlay bottom centre of the view (centred on the viewport the scene now renders into) | done | `ui/overlays.rs:41` |
| 146 | Space: play / pause | Unchanged; shown under TB ▸ Time | done | `main.rs:445`, `ui/toolbar.rs:304` |
| 147 | Delete: remove selected | Unchanged; shown in Selection tab and Help | done | `main.rs:435`, `ui/dock/selection.rs:54` |
| 148 | mouse orbit / pan / zoom | Unchanged, in the viewport's frame (pan scaled by its height); Help ▸ Controls | done | `main.rs:487`, `ui/menus.rs:620` |
| 149 | M5b: container walls on/off (**on by default** since 2026-10-14, at the slider's default 1.5 Bohr inside the face; `--no-walls` turns them off) | Scene ▸ Container (Q4 seam) | done | `ui/menus.rs:390`, `ui/menus.rs:397`, `ui/mod.rs` `walls_on`; tests `fresh_panels_start_with_walls_on_at_the_default_half_width`, `a_fresh_worker_runs_with_the_panels_walls` |
| 150 | M5b: container half-width slider and its Bohr/Å readout (follows a preset / scene switch to the new cell's default, fix round 1) | Scene ▸ Container | done | `ui/menus.rs:404`, `ui/menus.rs:412`, `ui/actions.rs` Preset/LoadScene |
| 151 | M5b: "the run's walls" line | Scene ▸ Container | done | `ui/menus.rs:420` |
| 152 | M5b: "of which walls (piston)" row | Readouts ▸ Energies, indented | done | `ui/dock/readouts.rs:116` |
| 153 | M5b: atomic superposition between DFT clouds | View ▸ Density ▸ Estimated solver's cloud (Q5 seam) | done | `ui/menus.rs:361`, `ui/menus.rs:367` |
| 154 | M5b: refresh cloud every N s (+ slider) | View ▸ Density ▸ Estimated solver's cloud | done | `ui/menus.rs:371` |
| 155 | M5b: "(cloud options apply to the estimated solver …)" note | View ▸ Density | done | `ui/menus.rs:378` |
| 156 | M5b: draft mode "guess instead of falling back (unsafe)" | Physics ▸ Estimated solver + Dock ▸ Model (Q1 seam) | done | `ui/menus.rs:386`, `ui/dock/model.rs:115` |
| 157 | M5b: draft mode's red panel line | Under the draft toggle (both places) | done | `ui/menus.rs:396` |
| 158 | M5b: red "extrapolating" banner | Overlay, top centre of the view (Q2 seam) | done | `ui/overlays.rs:70`, `viewer.rs:983` |
| 159 | M5b: atoms tinted by uncertainty | 3D spheres (Q3), unchanged | done | `gfx.rs:338` |
| 160 | M5b: draft guesses red in the sparkline + "draft guess" legend | Dock ▸ Graph + Readouts mini sparkline; status chip pill (new) | done | `ui/dock/graph.rs:140`, `ui/overlays.rs:182` |
| 161 | M5b: last audit "model off by … at step n" | Dock ▸ Model | done | `ui/dock/model.rs:347` |
| 162 | M5b: E conserved / work-done tooltips with the walls | Readouts ▸ Energies (main's text) | done | `ui/dock/readouts.rs:101` |
| 163 | M5b: smooth-motion tooltip with the crossfade | View ▸ Display aids (main's text) | done | `ui/menus.rs:432` |
| 164 | M5b: start-up flags --scene / --walls / --no-walls / --quality / --draft / --cloud-refresh | Unchanged (window, --screenshot and --replay); `--no-walls` and `--quality` added 2026-10-14 | done | `main.rs:140`, `screenshot.rs:138` |
| 165 | drag pull (new: the wheel while dragging sets the drag spring's force cap, ×1.25 per line, 0.01–1 Ha/Bohr) | Wheel during a Select-drag (zoom otherwise) → `Action::DragForceWheel`; "pull … Ha/Bohr · … eV/Å" label by the cursor (Overlay); TB ▸ Tool sub-line in Select mode; Physics ▸ Drag pull (log slider + Reset to 0.1); Ctrl K "pull" / "drag force"; Orbit / Select tooltips and Help ▸ Controls mention it | done | `main.rs:430`, `viewer.rs:501`, `viewer.rs:899`, `ui/toolbar.rs:357`, `ui/menus.rs:557`, `ui/search.rs:89` |
| 166 | M6 (Task 3 fix round 2): "bond references: O 2/3… (14 s)" — the isolated-atom references solving in the background (old panel: a weak line under the status line, never "solving…") | Overlay status chip, a dim line under the state (`overlays::chip_notes`, before the status line and the refusal) | done | `ui/overlays.rs:129`, `ui/overlays.rs:143` |
| 167 | M6 (Task 9): the sampling-pass estimate's basis — the long first-pass offer ("≈ t at this scene's first solve … a pass's own solves differ (references, warm starts) — the time left is shown once it runs", "Start anyway (≈ t at the first solve)"), the armed sampling button ("… at the last pass's rate" once `ModelStatus::solve_s_measured`, else "at the first solve") and its tooltip | Dock ▸ Model + the offer card in the view (`dock::offer`); Dock ▸ Model sampling row (`actions::sample_button`) | done | `ui/dock/model.rs:147`, `ui/dock/model.rs:155`, `ui/actions.rs:297`, `ui/dock/model.rs:168` |
| 168 | M6 (Task 5): the atom card's isotope — the title (²H) and the neutron count follow the nucleus's own mass number (`SceneNucleus::mass_number`, else the most abundant isotope) | Card + Dock ▸ Selection (`atom_card::title` / `composition` take the nucleus) | done | `atom_card.rs:22`, `atom_card.rs:51`, `ui/dock/selection.rs:18`, `ui/dock/selection.rs:28` |
| 169 | M6 Q6: bond lines between bonded nuclei (strand bundle; multi-centre ring bonds dashed violet), between the positions drawn now; nothing on a count change | Overlay (3D lines, depth-tested) | done | `overlay.rs:44`, `viewer.rs:1188` |
| 170 | M6 Q6: the stale bonds under the surrogate — faded to 0.3 and labelled "bonds from the last DFT step (t = … fs) — the surrogate has no electrons" | Overlay bottom centre, with the faded-density label; Dock ▸ Selection ▸ Bonds | done | `ui/mod.rs:694`, `viewer.rs:1201`, `ui/overlays.rs:40` |
| 171 | M6 Q6: molecule labels "H₂O — water" above each molecule of ≥ 2 nuclei (faded when stale), computed once per bond graph on the worker (`Snapshot::labels`, `charge_note`) | Overlay (egui foreground) | done | `viewer.rs:1239`, `worker/mod.rs:860`, `worker/mod.rs:524` |
| 172 | M6 Q6/Q7: View ▸ Bonds and molecules — bonds, molecule names, composition badge (on by default), the bond criterion's tooltip and the honesty lines | Menu ▸ View | done | `ui/menus.rs:449`, `ui/menus.rs:452`, `ui/menus.rs:458` |
| 173 | M6 Q6: the selected nucleus's bonds with each pair's numbers or rejection reason (`ui::pair_line`), its molecule; the Molecules list with the unassigned net charge and "H₃ is not a stable molecule; H₃⁺ is" | Dock ▸ Selection | done | `ui/dock/selection.rs:98`, `ui/dock/selection.rs:137`, `ui/mod.rs:661` |
| 174 | M6 Q7: composition badge "¹⁶O · 8 p · 8 n · \|F\| … Ha/Bohr" — hovered nucleus (Select tool) at the cursor, else the selection beside it | Overlay | done | `viewer.rs:1209` |
| 175 | M6 Q8: Relax / Stop (refused during a pass, Play and Step wait while it runs), the readout (steps, max\|F\| vs F_TOL, ΔE, FIRE Δt), the teacher-verification line and the driver's note | TB ▸ Time (full layout) + Menu ▸ Physics ▸ Relax geometry + a relax card over the view | done | `ui/mod.rs:621`, `ui/toolbar.rs:310`, `ui/menus.rs:595`, `ui/overlays.rs:247`, `ui/mod.rs:555`, `ui/actions.rs:285` |
| 176 | M6: the selected nucleus's isotope (listed A with masses; `SetIsotope`) | Dock ▸ Selection | done | `ui/dock/selection.rs:84`, `ui/mod.rs:638`, `ui/actions.rs:142` |
| 177 | M6 (user request 2026-10-14): periodic table — 18 columns + f rows from the period boundaries, search by symbol / name / Z, Z ≤ 10 enabled, the rest greyed "needs pseudopotentials (planned)"; the chosen element's isotopes (most abundant default) → the placement isotope (`AddNucleus { a }`) | TB ▸ Element "periodic table ›" (full) / the element chip (compact) — popover | done | `ui/ptable.rs:33`, `ui/ptable.rs:55`, `ui/ptable.rs:170`, `ui/ptable.rs:184`, `ui/toolbar.rs:389`, `ui/toolbar.rs:589`, `ui/actions.rs:135`, `viewer.rs:569` |
| 178 | M6 (Task 5 parked): the model's info line "taught at …, masses ¹H (recorded, not compared)" (`Provenance::describe_recorded`) | Dock ▸ Model | done | `ui/dock/model.rs:38` |
| 179 | M6: Help ▸ Controls rows for Relax, element and isotope, bonds and molecules | Menu ▸ Help | done | `ui/menus.rs:690`, `ui/menus.rs:691` |
| 180 | M6 (user request 2026-10-15): molecule name style — common / systematic (IR-5 stoichiometric: "dihydrogen monoxide") / IUPAC hydride ("oxidane") / formula only; the formula always shows; Ctrl K "name style" | Menu ▸ View ▸ Bonds and molecules (segmented); labels in the view and Dock ▸ Selection follow it | done | `ui/menus.rs:466`, `ui/actions.rs:284`, `names.rs:159`, `names.rs:205`, `names.rs:250`, `names.rs:279`, `names.rs:300` |
| 181 | M6 (user request 2026-10-15): sampling-plan presets — Quick look / Gentle / Standard (= `SamplingPlan::default()`) / Hot / reactive, each with its use case as hover text and under the row; "Custom" when the fields match none (derived from the plan, seed ignored); picking one keeps the seed; the row's hover text is the rule of thumb (sample at ~2× the play temperature); Ctrl K "preset" | Dock ▸ Model ▸ Sampling plan (ComboBox above the fields) | done | `ui/dock/model.rs:247`, `ui/sampling_presets.rs:25`, `ui/sampling_presets.rs:89`, `ui/search.rs:133` |
| 182 | M6 (user request 2026-10-15, "bonds that aren't true"): bond strength — the line's width and opacity follow log ρ_b (0.01 → 0.3 e/Bohr³, a display scale), its pattern the QTAIM kind (sign of ∇²ρ_b against 3× its estimated noise): solid shared, dashed closed-shell, dotted indeterminate; ring bonds keep violet but no longer dash (row 169's dashes now mean closed-shell); each bonded pair's line in the Selection tab adds "Laplacian … e/Bohr⁵: covalent (shared)" / "weak / stretched contact (closed-shell)" / "not resolved on this grid", its hover the explanation and "LSDA overbinds stretched contacts (delocalisation error): a weak contact here may be weaker or absent in reality" (also shown as a note whenever a bond is listed — review r1); View ▸ Bonds and molecules tooltip explains the styles | Overlay (3D lines); Dock ▸ Selection ▸ Bonds; Menu ▸ View ▸ Bonds and molecules (tooltip) | done | `overlay.rs:56`, `overlay.rs:62`, `overlay.rs:73`, `ui/mod.rs:758`, `ui/mod.rs:761`, `ui/mod.rs:778`, `ui/dock/selection.rs:120`, `ui/menus.rs:550` |

## Deviations from the mockup's wording (coordinator rulings, fix round 1)

- **Rows 5 and 6, refusals and status lines (review I2).** The mockup moves them to toasts plus the Log; the old panel kept them in place. Both are in place again: the status chip shows the status line until it is replaced or cleared (amber when it asks for something, e.g. "paused: … press Play to continue"; Play clears it) and the driver's refusal until the next accepted command clears it; the Log tab also has a last refusal line. A repeated message gets fresh feedback: UiState::set_status bumps status_seq and a refusal is keyed on (text, Snapshot::edits), so Save twice or Space twice during a pass re-toast and re-log. Refusal toasts sit above the toolbar (bottom centre), status toasts at the bottom right.
- **Row 72, dynamics notes.** Readouts ▸ Health shows the three newest notes, as the old panel did.
  The mockup says "newest". Ruling: keep more information, not less. All notes are also in the Log.

## Fix round 1: the 3D view renders into the central viewport

The scene used to render full-window under the panels, which put it off-centre. It now renders into
the rect the top bar, toolbar and dock leave. One `render::camera::Viewport { origin, size }`
(physical pixels) is computed per frame (`viewer.rs` `viewport_of`). It drives:
- the render pass viewport and scissor (`Renderer::render_in`);
- the camera aspect and the shaders' pixel → NDC mapping (`CameraUniform::origin`);
- every mouse ↔ world path: `tools::pick`, `place_point`, `drag_point` and `project`, the
  distance guides, the drag spring and the pin squares (clipped to the view), and pan scaling.

Presses outside the viewport are ignored. The `--screenshot` frame is still the full window
(chrome + viewport).

## Not in Appendix A, new in this branch (approved conveniences)

- Ctrl K control search (`ui/search.rs`): fuzzy match over every control's name; picking a result
  opens its menu or dock tab (and collapsible), scrolls to it and outlines it.
- Ha / eV / both toggle for the Readouts energies (`ui/dock/mod.rs`, `ui/mod.rs` `energy_cells`).
- Log tab with filters and the strip's unread error / refusal counter (`ui/logbook.rs`,
  `ui/dock/log.rs`); toasts for status lines and refusals (`ui/overlays.rs` `toast`).
- Picking an element chip switches the tool to Place (`ui/actions.rs` `Action::Element`).
- Esc deselects (`ui/mod.rs` `keys`; not while a text field, a menu or the search has it).
- The atom card can be hidden (Selection tab).
- Volume / Isosurface quick switch and the axes gizmo (`ui/overlays.rs` `view_switch`, `gizmo`).
- Dock strip peeks (hover a tab button for its key numbers) and status dots.
- Help windows with the long-form "Physics, honestly" texts (`ui/menus.rs` `help_text`).

## Appendix B items

Q1–Q5 arrived with the M5b merge from `main` and are real controls now (rows 149–164); Q6–Q8
with M6 Task 10 (rows 169–175). No `QUEUED` placeholder is left.

| # | Item | Where |
|---|---|---|
| Q1 | draft mode toggle | **real**: `ui/menus.rs` `draft_control` (Physics ▸ Estimated solver; also Dock ▸ Model) |
| Q2 | "extrapolating" banner | **real**: `ui/overlays.rs` `draft_banner` (top centre of the view) |
| Q3 | per-atom uncertainty tint | **real**: `gfx::risk_tint` of `viewer::Viewer::atom_risk` (sphere colour) |
| Q4 | container walls + size | **real**: `ui/menus.rs` `walls_controls` (Scene ▸ Container) |
| Q5 | live-cloud refresh | **real**: `ui/menus.rs` `cloud_controls` (View ▸ Density) |
| Q6 | bond lines and molecule names | **real**: `overlay::bond_lines`, `Viewer::label_overlay`, View ▸ Bonds and molecules, Dock ▸ Selection (rows 169–173) |
| Q7 | composition badge | **real**: `Viewer::badge` (row 174) |
| Q8 | relax (FIRE) | **real**: TB ▸ Time and Physics ▸ Relax geometry, the relax card (row 175) |
