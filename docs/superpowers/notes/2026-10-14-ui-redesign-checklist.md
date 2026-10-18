# UI redesign: Appendix A checklist (2026-10-14)

The acceptance checklist of the UI redesign: every row of Appendix A of the mockup
(`docs/superpowers/notes/2026-10-14-ui-mockup.html`, 148 controls and readouts of the old side
panel, the atom card, the overlays and the mouse/keyboard input) with where it lives now. File
references are relative to `crates/app/src/` and point at the line that draws (or records) the
item. Every line number was **re-anchored at the M6 final fix wave (2026-10-15, final review app
M6)**: each anchor's text at the commit that last wrote its row was looked up in the fixed tree
(the occurrence nearest the old line), and the 30 anchors whose old text was already stale or
trivial (`}`) were placed by hand; then every anchored line was read against its row. Later
edits move line numbers again — the cited line's text, not its number, is the anchor. Tooltip
texts of the old panel are kept verbatim on the moved controls.

**194 / 194 rows done** (148 from Appendix A, then the M5b viewer extras merged from `main` as rows 149–164, the drag pull — user request 2026-10-14, branch `drag-force-wheel` — as row 165, the M6 branch's edits to the old `ui.rs`, ported when `main` was merged into `m6-tools`, as rows 166–168, and the M6 Task 10 controls — Appendix B's Q6–Q8 and the periodic table — as rows 169–179, and the molecule name style — user request 2026-10-15 — as row 180, and the sampling-plan presets — user request 2026-10-15, branch `m6-sampling-presets` — as row 181, and the bond-strength display — user request 2026-10-15, branch `m6-bond-strength` — as row 182, and the orbital viewer's controls — plan `2026-10-15-orbital-viewer.md` Task 7, branch `orbital-viewer` — as rows 183–192). All line numbers are as of the M6 final fix wave (header above). Rows 1, 80 and 81 (the scene chip, the preset picker, Reset) hold for a scene loaded from a file too since that fix wave (final review app I1: the chip names the file, no preset is highlighted, Reset reloads the file). The second merge of `main` into `m6-tools` (2026-10-14) brought main's two new rows, which main had numbered in place as **75a** (temperature scale K / °C / °F) and **81a** (Scene ▸ Quality): they are kept under those numbers (renumbering them after 180 would separate them from the controls they extend; no number overlaps 166–180), and main's edits to rows 80–81 (the `empty` preset), 149–151 (walls on by default) and 164 (`--no-walls`) replace the old text of those rows; their line numbers are as of that merge.

Abbreviations as in the mockup: TB = bottom toolbar, Dock ▸ X = right dock tab, Menu = top menu,
Overlay = drawn in the 3D view.

| # | Old item | New place | Status | Where (file:line) |
|---|---|---|---|---|
| 1 | heading "Atomic Sandbox" | Top bar brand + scene chip | done | `ui/menus.rs:93`, `ui/menus.rs:104` |
| 2 | status line + spinner | Overlay status chip (all states kept; busy states worded by the activity headline, activity spec §3.1) | done | `ui/overlays.rs:164`, `ui/overlays.rs:186` |
| 3 | UI error (red) | Status chip red + Log (error) | done | `ui/overlays.rs:180`, `ui/logbook.rs:169` |
| 4 | "paused by a solver failure" | Status chip red + Log + strip count | done | `ui/overlays.rs:166`, `ui/logbook.rs:180` |
| 5 | "refused: …" (amber) | Status chip (amber line) and Log tab "last refusal" line, in place while the driver holds it + toast above the toolbar (re-raised per attempt) + Log entry | done | `ui/overlays.rs:146`, `ui/dock/log.rs:10`, `ui/logbook.rs:187`, `ui/overlays.rs:300` |
| 6 | neutral status (saved / loaded / switched; "paused: … press Play to continue") | Status chip line until replaced or cleared (amber when it asks for something) + toast per set_status + Log (status) | done | `ui/overlays.rs:143`, `ui/mod.rs:446`, `ui/logbook.rs:166`, `ui/overlays.rs:279` |
| 7 | solver radio · CPU | TB Solver segmented | done | `ui/toolbar.rs:626` |
| 8 | solver radio · GPU | TB Solver segmented, "no GPU adapter" | done | `ui/toolbar.rs:631` |
| 9 | solver radio · estimated | TB Solver segmented, tooltip names the teacher | done | `ui/toolbar.rs:38` |
| 10 | honesty line | Dock ▸ Model first line + i; Help ▸ Physics, honestly | done | `ui/dock/model.rs:19`, `ui/menus.rs:807` |
| 11 | this step (surrogate / fallback / DFT) | TB Solver sub-line + status chip pill + Dock ▸ Model | done | `ui/dock/model.rs:341`, `ui/toolbar.rs:649`, `ui/overlays.rs:194` |
| 12 | uncertainty vs bound | Dock ▸ Model (bar + numbers, tooltip kept) | done | `ui/dock/model.rs:344`, `ui/dock/model.rs:455` |
| 13 | trusted (five variants) | TB trust light + Dock ▸ Model + strip dot (the text is the Activity checklist's first unmet line, the fit's own words on hover; activity Task 8) | done | `ui/mod.rs:981`, `ui/dock/model.rs:361`, `ui/dock/mod.rs:59` |
| 14 | samples (+ refit pending) | Dock ▸ Model | done | `ui/dock/model.rs:368` |
| 15 | model epoch | Dock ▸ Model + TB sub-line | done | `ui/dock/model.rs:371`, `ui/toolbar.rs:683` |
| 16 | last fit | Dock ▸ Model | done | `ui/dock/model.rs:374` |
| 17 | last refit | Dock ▸ Model | done | `ui/dock/model.rs:398` |
| 18 | fallbacks % of last 100 | Dock ▸ Model (bar) + TB sub-line | done | `ui/dock/model.rs:405`, `ui/toolbar.rs:683` |
| 19 | teacher gap | Dock ▸ Model | done | `ui/dock/model.rs:416` |
| 20 | last audit (verdict, gap, ms, lag, in flight) | Dock ▸ Model; audit lag also in Readouts | done | `ui/dock/model.rs:424`, `ui/dock/readouts.rs:177` |
| 21 | mismatch work | Dock ▸ Model | done | `ui/dock/model.rs:446` |
| 22 | provenance mismatch warning | Dock ▸ Model warning slot + Log | done | `ui/dock/model.rs:46`, `ui/logbook.rs:244` |
| 23 | autoload refused warning | Dock ▸ Model warning slot + Log | done | `ui/dock/model.rs:49`, `ui/logbook.rs:251` |
| 24 | "Forget it and sample this scene" | Dock ▸ Model, in the warning | done | `ui/dock/model.rs:50` |
| 25 | sampling progress bar | Sampling card + status chip + Dock ▸ Model | done | `ui/dock/model.rs:144`, `ui/overlays.rs:242` |
| 26 | last teacher energy · skipped · failed | Sampling card + Dock ▸ Model | done | `ui/dock/model.rs:150` |
| 27 | last pass summary | Dock ▸ Model + Log (pass) | done | `ui/dock/model.rs:63`, `ui/logbook.rs:226` |
| 28 | long first-pass offer (amber) | Offer card in the view + Dock ▸ Model | done | `ui/dock/model.rs:156`, `ui/overlays.rs:27` |
| 29 | offer "Start anyway (≈ t)" | Offer card (and Model tab) | done | `ui/dock/model.rs:169` |
| 30 | DFT audit every n | Dock ▸ Model ▸ Audits | done | `ui/dock/model.rs:79` |
| 31 | "(solver: n)" | Dock ▸ Model, beside the field | done | `ui/dock/model.rs:91` |
| 32 | audits-off unsafe warning | Dock ▸ Model, under the field | done | `ui/dock/model.rs:95` |
| 33 | plan · perturbations | Dock ▸ Model ▸ Sampling plan | done | `ui/dock/model.rs:276` |
| 34 | plan · σ_r | Dock ▸ Model ▸ Sampling plan | done | `ui/dock/model.rs:280` |
| 35 | plan · hot burst steps | Dock ▸ Model ▸ Sampling plan | done | `ui/dock/model.rs:286` |
| 36 | plan · hot burst T | Dock ▸ Model ▸ Sampling plan | done | `ui/dock/model.rs:289` |
| 37 | plan · scan points / pair | Dock ▸ Model ▸ Sampling plan | done | `ui/dock/model.rs:296` |
| 38 | plan · seed | Dock ▸ Model ▸ Sampling plan | done | `ui/dock/model.rs:302` |
| 39 | Start sampling / Sample more / armed "Start anyway" | Dock ▸ Model primary button (two clicks above 2 min) | done | `ui/dock/model.rs:174`, `ui/actions.rs:664` |
| 40 | plan cost "n solves, ≈ t" | Dock ▸ Model, beside the button | done | `ui/actions.rs:682` |
| 41 | Cancel | Sampling card + Dock ▸ Model | done | `ui/dock/model.rs:199`, `ui/overlays.rs:237` |
| 42 | Forget model (two clicks within 4 s) | File ▸ Estimated model + Dock ▸ Model | done | `ui/dock/model.rs:234`, `ui/menus.rs:238` |
| 43 | Save model | File ▸ Estimated model + Dock ▸ Model | done | `ui/dock/model.rs:215` |
| 44 | Load model | File ▸ Estimated model + Dock ▸ Model | done | `ui/dock/model.rs:222` |
| 45 | model path label | File ▸ Estimated model header + Dock ▸ Model | done | `ui/menus.rs:235`, `ui/dock/model.rs:114` |
| 46 | Play / Pause | TB ▸ Time primary button (Space; sampling reason kept) | done | `ui/toolbar.rs:501` |
| 47 | Step | TB ▸ Time | done | `ui/toolbar.rs:524` |
| 48 | Rewind | TB ▸ Time | done | `ui/toolbar.rs:505` |
| 49 | speed slider (steps/tick, log) | TB ▸ Speed (slider + typed field) | done | `ui/toolbar.rs:327` |
| 50 | "watchable (≈ fs/s)" | TB ▸ Speed sub-line | done | `ui/toolbar.rs:345` |
| 51 | measured fs/s (last 2 s) | TB ▸ Speed (large) + Readouts tile and Time and cost | done | `ui/toolbar.rs:337`, `ui/dock/readouts.rs:77` |
| 52 | fs/s this tick | TB ▸ Speed sub-line + Readouts | done | `ui/toolbar.rs:351`, `ui/dock/readouts.rs:152` |
| 53 | steps last tick | Readouts ▸ Time and cost | done | `ui/dock/readouts.rs:154` |
| 54 | ms / step | Readouts ▸ Time and cost | done | `ui/dock/readouts.rs:156` |
| 55 | dt | TB ▸ dt + Physics menu | done | `ui/menus.rs:569`, `ui/toolbar.rs:360` |
| 56 | t = fs, step N | Status chip + Readouts | done | `ui/overlays.rs:202`, `ui/dock/readouts.rs:147` |
| 57 | dt warning (amber) | TB dt field amber + Readouts ▸ Health + Log | done | `ui/menus.rs:571`, `ui/dock/readouts.rs:196` |
| 58 | T | Readouts tile + status chip | done | `ui/dock/readouts.rs:48`, `ui/overlays.rs:205` |
| 59 | KE (nuclei) | Readouts ▸ Energies (Ha / eV / both) | done | `ui/dock/readouts.rs:95` |
| 60 | E_KS | Readouts ▸ Energies | done | `ui/dock/readouts.rs:96` |
| 61 | E total | Readouts ▸ Energies (bold) | done | `ui/dock/readouts.rs:97` |
| 62 | E conserved | Readouts ▸ Energies | done | `ui/dock/readouts.rs:99` |
| 63 | work done by tools | Readouts ▸ Energies | done | `ui/dock/readouts.rs:106` |
| 64 | model refit (estimated) | Readouts ▸ Energies, indented | done | `ui/dock/readouts.rs:125` |
| 65 | model force mismatch (estimated) | Readouts ▸ Energies, indented | done | `ui/dock/readouts.rs:131` |
| 66 | drift (coloured) | Readouts tile + strip dot (1e-4 / 1e-3) | done | `ui/dock/readouts.rs:65`, `ui/dock/mod.rs:50` |
| 67 | per step (model ms / SCF steps, readbacks, ms) | Readouts ▸ Time and cost | done | `ui/dock/readouts.rs:162` |
| 68 | sparkline (E total by source, T, fallback dots) | Dock ▸ Graph (large) + mini in Readouts | done | `ui/dock/graph.rs:45`, `ui/dock/readouts.rs:189` |
| 69 | sparkline ranges + "last N fs" | Dock ▸ Graph | done | `ui/dock/graph.rs:130` |
| 70 | source legend | Dock ▸ Graph | done | `ui/dock/graph.rs:136` |
| 71 | box warning (amber / red) | Readouts ▸ Health + Log + strip dot | done | `ui/dock/readouts.rs:192`, `ui/logbook.rs:192` |
| 72 | dynamics notes (last 3) | Readouts ▸ Health (the 3 newest, as before) + Log (all) | done | `ui/dock/readouts.rs:203`, `ui/logbook.rs:99` |
| 73 | thermostat on/off | Physics menu + status chip "CSVR T K" | done | `ui/menus.rs:604`, `ui/overlays.rs:207` |
| 74 | thermostat target T | Physics menu | done | `ui/menus.rs:612` |
| 75 | thermostat τ | Physics menu | done | `ui/menus.rs:621` |
| 75a | temperature scale K / °C / °F (display only; user request 2026-10-14): T readout, status chip, sparkline range, thermostat target, hot-burst T; typed values clamped at absolute zero | Physics ▸ Thermostat (segmented, like Ha/eV/both); Ctrl K "celsius" / "fahrenheit" | done | `ui/menus.rs:597`, `ui/menus.rs:610`, `ui/mod.rs` `TempUnits`, `units` conversions; tests `temperatures_show_in_the_chosen_scale_and_stay_kelvin_inside`, `the_thermostat_target_is_typed_in_the_chosen_scale` |
| 76 | gravity "g = 9.8 m/s² (−y)" | Physics menu | done | `ui/menus.rs:639` |
| 77 | gravity "off" | Physics menu | done | `ui/menus.rs:636` |
| 78 | COM correction | Physics menu | done | `ui/menus.rs:665` |
| 79 | field magnitude readout | Physics menu, under the field | done | `ui/menus.rs:655` |
| 80 | preset combo (empty, h, h2, he, lih, h2o — `empty` added 2026-10-14) | File ▸ Presets + scene chip | done | `ui/menus.rs:178`, `ui/menus.rs:138`; test `the_presets_popup_lists_empty_first_and_sends_its_scene` |
| 81 | Reset | File ▸ Presets + scene chip | done | `ui/menus.rs:192`, `ui/menus.rs:197`; test `a_loaded_scene_file_is_the_chips_source_and_reset_reloads_it` |
| 81a | Quality: Fast / Balanced / Accurate (default) / Precise + Custom h, with grid, ≈ cost and H₂ accuracy per level (user request 2026-10-14) | Scene ▸ Quality; Ctrl K "Quality" | done | `ui/menus.rs` `quality_controls`, `quality.rs`; tests `clicking_a_quality_level_sends_the_grid_edit`, `quality_changes_keep_the_box_and_leave_sigma_alone` |
| 82 | tool Orbit | TB ▸ Tool | done | `ui/toolbar.rs:542` |
| 83 | tool Place | TB ▸ Tool | done | `ui/toolbar.rs:543` |
| 84 | tool Select | TB ▸ Tool | done | `ui/toolbar.rs:544` |
| 85 | element to place (H–Ne) | TB ▸ element chips (pick → Place); compact flyout | done | `ui/toolbar.rs:560`, `ui/toolbar.rs:587`, `ui/actions.rs:145` |
| 86 | "no selection" hint | TB ▸ Selection placeholder + Dock ▸ Selection | done | `ui/toolbar.rs:407`, `ui/dock/selection.rs:14` |
| 87 | "selected: X #i at (x, y, z) Bohr" | TB ▸ Selection label + Dock ▸ Selection | done | `ui/toolbar.rs:397`, `ui/dock/selection.rs:27` |
| 88 | Delete selected | TB ▸ Selection + Dock ▸ Selection | done | `ui/toolbar.rs:613`, `ui/dock/selection.rs:51` |
| 89 | Focus selected | TB ▸ Selection + Dock ▸ Selection | done | `ui/toolbar.rs:605`, `ui/dock/selection.rs:48` |
| 90 | pinned | TB ▸ Selection Pin + Dock ▸ Selection | done | `ui/toolbar.rs:597`, `ui/dock/selection.rs:42` |
| 91 | hint "… edits pause the run and restart its time" | Scene menu, top line | done | `ui/menus.rs:249` |
| 92 | charge | Scene ▸ Electrons + scene chip | done | `ui/menus.rs:253` |
| 93 | multiplicity auto | Scene ▸ Electrons | done | `ui/menus.rs:264` |
| 94 | multiplicity value | Scene ▸ Electrons | done | `ui/menus.rs:265` |
| 95 | σ auto = h | Scene ▸ Electrons | done | `ui/menus.rs:275` |
| 96 | σ value | Scene ▸ Electrons | done | `ui/menus.rs:277` |
| 97 | grid n | Scene ▸ Grid (odd n only) | done | `ui/menus.rs:294` |
| 98 | grid h | Scene ▸ Grid | done | `ui/menus.rs:304` |
| 99 | cost readout | Scene ▸ Grid | done | `ui/menus.rs:313` |
| 100 | Apply grid | Scene ▸ Grid (enabled with a pending edit) | done | `ui/menus.rs:306` |
| 101 | density volume / isosurface | View ▸ Density + quick switch in the view | done | `ui/menus.rs:487`, `ui/overlays.rs:333` |
| 102 | gamma | View ▸ Density (disabled for isosurface) | done | `ui/menus.rs:493` |
| 103 | opacity | View ▸ Density (disabled for isosurface) | done | `ui/menus.rs:494` |
| 104 | iso (× ρ max) | View ▸ Density (disabled for volume) | done | `ui/menus.rs:495` |
| 105 | cell | View ▸ Display aids | done | `ui/menus.rs:504` |
| 106 | floor grid | View ▸ Display aids | done | `ui/menus.rs:505` |
| 107 | axes | View ▸ Display aids | done | `ui/menus.rs:506` |
| 108 | smooth motion | View ▸ Display aids | done | `ui/menus.rs:516` |
| 109 | placement plane | View ▸ Display aids | done | `ui/menus.rs:509` |
| 110 | spacing 1 Bohr / 1 Å | View ▸ Display aids | done | `ui/menus.rs:512` |
| 111 | E_KS (breakdown) | Readouts ▸ Energy breakdown and solver (collapsible) | done | `ui/dock/readouts.rs:237` |
| 112 | kinetic (electrons) | same collapsible | done | `ui/dock/readouts.rs:238` |
| 113 | external | same collapsible | done | `ui/dock/readouts.rs:239` |
| 114 | Hartree | same collapsible | done | `ui/dock/readouts.rs:240` |
| 115 | xc | same collapsible | done | `ui/dock/readouts.rs:241` |
| 116 | nuclear | same collapsible | done | `ui/dock/readouts.rs:242` |
| 117 | note "smeared nuclei (σ = …)" | same collapsible | done | `ui/dock/readouts.rs:245` |
| 118 | surrogate / fallback breakdown note | same collapsible | done | `ui/dock/readouts.rs:247` |
| 119 | GPU f32 scatter note | same collapsible | done | `ui/dock/readouts.rs:252` |
| 120 | SCF steps | same collapsible | done | `ui/dock/readouts.rs:260` |
| 121 | eigen residual (final) | same collapsible | done | `ui/dock/readouts.rs:263` |
| 122 | Poisson residual | same collapsible | done | `ui/dock/readouts.rs:264` |
| 123 | solve (ms, readbacks, engine) | same collapsible | done | `ui/dock/readouts.rs:268` |
| 124 | grid | same collapsible | done | `ui/dock/readouts.rs:272` |
| 125 | ρ max | same collapsible | done | `ui/dock/readouts.rs:276` |
| 126 | evaluation notes | same collapsible + Log | done | `ui/dock/readouts.rs:279`, `ui/logbook.rs:219` |
| 127 | evaluation warnings (amber) | Readouts ▸ Health + Log | done | `ui/dock/readouts.rs:200`, `ui/logbook.rs:219` |
| 128 | scene path field | File ▸ Scene file | done | `ui/menus.rs:212` |
| 129 | Save | File ▸ Scene file | done | `ui/menus.rs:214` |
| 130 | Load (+ .model sidecar) | File ▸ Scene file (sidecar rule and "sampling was running" kept) | done | `ui/menus.rs:221`, `ui/actions.rs:266` |
| 131 | atom card title | Overlay card top-right of the view + Dock ▸ Selection; card toggle (new) | done | `atom_card.rs:42`, `ui/dock/selection.rs:176` |
| 132 | "nucleus #i: p protons · n neutrons" | Card + Dock ▸ Selection | done | `atom_card.rs:90`, `ui/dock/selection.rs:30` |
| 133 | "z electrons (neutral atom), shells …" | Card + Dock ▸ Selection | done | `atom_card.rs:58`, `ui/dock/selection.rs:33` |
| 134 | Bohr schematic drawing | Card | done | `atom_card.rs:93` |
| 135 | "schematic — not to scale" | Card | done | `atom_card.rs:126` |
| 136 | shared-electrons caveat | Card (line + i) + Selection tab + Help | done | `atom_card.rs:67`, `ui/dock/selection.rs:177` |
| 137 | cell box (amber wire) | Overlay, unchanged (toggle #105) | done | `overlay.rs:129` |
| 138 | floor grid | Overlay, unchanged (toggle #106) | done | `overlay.rs:136` |
| 139 | axes | Overlay + corner gizmo (new) (toggle #107) | done | `overlay.rs:141`, `ui/overlays.rs:325` |
| 140 | placement plane (blue) | Overlay in Place mode (toggle #109) | done | `overlay.rs:144` |
| 141 | distance guides | Overlay (through the frame's viewport, clipped to it) + list in Dock ▸ Selection | done | `viewer.rs:1186`, `ui/dock/selection.rs:62` |
| 142 | placement marker cross | Overlay | done | `viewer.rs:1192` |
| 143 | drag spring ring + tether | Overlay | done | `viewer.rs:1147` |
| 144 | pin square | Overlay | done | `viewer.rs:1135` |
| 145 | faded-density label | Overlay bottom centre of the view (centred on the viewport the scene now renders into) | done | `ui/overlays.rs:45` |
| 146 | Space: play / pause | Unchanged; shown under TB ▸ Time | done | `viewer.rs:564`, `ui/toolbar.rs:318` |
| 147 | Delete: remove selected | Unchanged; shown in Selection tab and Help | done | `viewer.rs:559`, `ui/dock/selection.rs:58` |
| 148 | mouse orbit / pan / zoom | Unchanged, in the viewport's frame (pan scaled by its height); Help ▸ Controls | done | `viewer.rs:620`, `ui/menus.rs:780` |
| 149 | M5b: container walls on/off (**on by default** since 2026-10-14, at the slider's default 1.5 Bohr inside the face; `--no-walls` turns them off) | Scene ▸ Container (Q4 seam) | done | `ui/menus.rs:403`, `ui/menus.rs:410`, `ui/mod.rs` `walls_on`; tests `fresh_panels_start_with_walls_on_at_the_default_half_width`, `a_fresh_worker_runs_with_the_panels_walls` |
| 150 | M5b: container half-width slider and its Bohr/Å readout (follows a preset / scene switch to the new cell's default, fix round 1) | Scene ▸ Container | done | `ui/menus.rs:417`, `ui/menus.rs:425`, `ui/actions.rs` Preset/LoadScene |
| 151 | M5b: "the run's walls" line | Scene ▸ Container | done | `ui/menus.rs:433` |
| 152 | M5b: "of which walls (piston)" row | Readouts ▸ Energies, indented | done | `ui/dock/readouts.rs:117` |
| 153 | M5b: atomic superposition between DFT clouds | View ▸ Density ▸ Estimated solver's cloud (Q5 seam) | done | `ui/menus.rs:444`, `ui/menus.rs:450` |
| 154 | M5b: refresh cloud every N s (+ slider) | View ▸ Density ▸ Estimated solver's cloud | done | `ui/menus.rs:454` |
| 155 | M5b: "(cloud options apply to the estimated solver …)" note | View ▸ Density | done | `ui/menus.rs:461` |
| 156 | M5b: draft mode "guess instead of falling back (unsafe)" | Physics ▸ Estimated solver + Dock ▸ Model (Q1 seam) | done | `ui/menus.rs:469`, `ui/dock/model.rs:128` |
| 157 | M5b: draft mode's red panel line | Under the draft toggle (both places) | done | `ui/menus.rs:479` |
| 158 | M5b: red "extrapolating" banner | Overlay, top centre of the view (Q2 seam) | done | `ui/overlays.rs:77`, `viewer.rs:1349` |
| 159 | M5b: atoms tinted by uncertainty | 3D spheres (Q3), unchanged | done | `gfx.rs:350` |
| 160 | M5b: draft guesses red in the sparkline + "draft guess" legend | Dock ▸ Graph + Readouts mini sparkline; status chip pill (new) | done | `ui/dock/graph.rs:141`, `ui/overlays.rs:196` |
| 161 | M5b: last audit "model off by … at step n" | Dock ▸ Model | done | `ui/dock/model.rs:436` |
| 162 | M5b: E conserved / work-done tooltips with the walls | Readouts ▸ Energies (main's text) | done | `ui/dock/readouts.rs:102` |
| 163 | M5b: smooth-motion tooltip with the crossfade | View ▸ Display aids (main's text) | done | `ui/menus.rs:517` |
| 164 | M5b: start-up flags --scene / --walls / --no-walls / --quality / --draft / --cloud-refresh | Unchanged (window, --screenshot and --replay); `--no-walls` and `--quality` added 2026-10-14 | done | `main.rs:140`, `screenshot.rs:280` |
| 165 | drag pull (new: the wheel while dragging sets the drag spring's force cap, ×1.25 per line, 0.01–1 Ha/Bohr) | Wheel during a Select-drag (zoom otherwise) → `Action::DragForceWheel`; "pull … Ha/Bohr · … eV/Å" label by the cursor (Overlay); TB ▸ Tool sub-line in Select mode; Physics ▸ Drag pull (log slider + Reset to 0.1); Ctrl K "pull" / "drag force"; Orbit / Select tooltips and Help ▸ Controls mention it | done | `viewer.rs:558`, `viewer.rs:740`, `viewer.rs:1162`, `ui/toolbar.rs:372`, `ui/menus.rs:717`, `ui/search.rs:96` |
| 166 | M6 (Task 3 fix round 2): "bond references: O 2/3… (14 s)" — the isolated-atom references solving in the background (old panel: a weak line under the status line, never "solving…") | Overlay status chip, a dim line under the state (`overlays::chip_notes`, before the status line and the refusal) | done | `ui/overlays.rs:139`, `ui/overlays.rs:153` |
| 167 | M6 (Task 9): the sampling-pass estimate's basis — the long first-pass offer ("≈ t at this scene's first solve … a pass's own solves differ (references, warm starts) — the time left is shown once it runs", "Start anyway (≈ t at the first solve)"), the armed sampling button ("… at the last pass's rate" once `ModelStatus::solve_s_measured`, else "at the first solve") and its tooltip | Dock ▸ Model + the offer card in the view (`dock::offer`); Dock ▸ Model sampling row (`actions::sample_button`) | done | `ui/dock/model.rs:160`, `ui/dock/model.rs:168`, `ui/actions.rs:674`, `ui/dock/model.rs:184` |
| 168 | M6 (Task 5): the atom card's isotope — the title (²H) and the neutron count follow the nucleus's own mass number (`SceneNucleus::mass_number`, else the most abundant isotope) | Card + Dock ▸ Selection (`atom_card::title` / `composition` take the nucleus) | done | `atom_card.rs:22`, `atom_card.rs:51`, `ui/dock/selection.rs:21`, `ui/dock/selection.rs:31` |
| 169 | M6 Q6: bond lines between bonded nuclei (strand bundle; multi-centre ring bonds dashed violet), between the positions drawn now; nothing on a count change | Overlay (3D lines, depth-tested) | done | `overlay.rs:73`, `viewer.rs:1224` |
| 170 | M6 Q6: the stale bonds under the surrogate — faded to 0.3 and labelled "bonds from the last DFT step (t = … fs) — the surrogate has no electrons" | Overlay bottom centre, with the faded-density label; Dock ▸ Selection ▸ Bonds | done | `ui/mod.rs:857`, `viewer.rs:1237`, `ui/overlays.rs:40` |
| 171 | M6 Q6: molecule labels "H₂O — water" above each molecule of ≥ 2 nuclei (faded when stale), computed once per bond graph on the worker (`Snapshot::labels`, `charge_note`) | Overlay (egui foreground) | done | `viewer.rs:1275`, `worker/mod.rs:878`, `worker/mod.rs:538` |
| 172 | M6 Q6/Q7: View ▸ Bonds and molecules — bonds, molecule names, composition badge (on by default), the bond criterion's tooltip and the honesty lines | Menu ▸ View | done | `ui/menus.rs:532`, `ui/menus.rs:535`, `ui/menus.rs:541` |
| 173 | M6 Q6: the selected nucleus's bonds with each pair's numbers or rejection reason (`ui::pair_line`), its molecule; the Molecules list with the unassigned net charge and "H₃ is not a stable molecule; H₃⁺ is" | Dock ▸ Selection | done | `ui/dock/selection.rs:98`, `ui/dock/selection.rs:151`, `ui/mod.rs:821` |
| 174 | M6 Q7: composition badge "¹⁶O · 8 p · 8 n · \|F\| … Ha/Bohr" — hovered nucleus (Select tool) at the cursor, else the selection beside it | Overlay | done | `viewer.rs:1245` |
| 175 | M6 Q8: Relax / Stop (refused during a pass, Play and Step wait while it runs), the readout (steps, max\|F\| vs F_TOL, ΔE, FIRE Δt), the teacher-verification line and the driver's note | TB ▸ Time (full layout) + Menu ▸ Physics ▸ Relax geometry + a relax card over the view | done | `ui/mod.rs:763`, `ui/toolbar.rs:310`, `ui/menus.rs:694`, `ui/overlays.rs:250`, `ui/mod.rs:697`, `ui/actions.rs:338` |
| 176 | M6: the selected nucleus's isotope (listed A with masses; `SetIsotope`) | Dock ▸ Selection | done | `ui/dock/selection.rs:84`, `ui/mod.rs:780`, `ui/actions.rs:154` |
| 177 | M6 (user request 2026-10-14): periodic table — 18 columns + f rows from the period boundaries, search by symbol / name / Z, Z ≤ 18 enabled (M7 Task 12: Z 11–18 tooltips state the pseudopotential outcome or the refusal and the all-electron fallback's Zσ), the rest greyed "needs pseudopotentials (planned)"; the chosen element's isotopes (most abundant default) → the placement isotope (`AddNucleus { a }`) | TB ▸ Element "periodic table ›" (full) / the element chip (compact) — popover | done | `ui/ptable.rs:33`, `ui/ptable.rs:55`, `ui/ptable.rs:170`, `ui/ptable.rs:184`, `ui/toolbar.rs:389`, `ui/toolbar.rs:589`, `ui/actions.rs:147`, `viewer.rs:594` |
| 178 | M6 (Task 5 parked): the model's info line "taught at …, masses ¹H (recorded, not compared)" (`Provenance::describe_recorded`) | Dock ▸ Model | done | `ui/dock/model.rs:42` |
| 179 | M6: Help ▸ Controls rows for Relax, element and isotope, bonds and molecules | Menu ▸ Help | done | `ui/menus.rs:789`, `ui/menus.rs:790` |
| 180 | M6 (user request 2026-10-15): molecule name style — common / systematic (IR-5 stoichiometric: "dihydrogen monoxide") / IUPAC hydride ("oxidane") / formula only; the formula always shows; Ctrl K "name style" | Menu ▸ View ▸ Bonds and molecules (segmented); labels in the view and Dock ▸ Selection follow it | done | `ui/menus.rs:549`, `ui/actions.rs:337`, `names.rs:159`, `names.rs:205`, `names.rs:250`, `names.rs:279`, `names.rs:300` |
| 181 | M6 (user request 2026-10-15): sampling-plan presets — Quick look / Gentle / Standard (= `SamplingPlan::default()`) / Hot / reactive, each with its use case as hover text and under the row (final review core I1: the copy promises only what the pass samples — no trust claim, far-stretched or unseen arrangements still DFT); "Custom" when the fields match none (derived from the plan, seed ignored); picking one keeps the seed; the row's hover text is the rule of thumb (sample at ~2× the play temperature); Ctrl K "preset" | Dock ▸ Model ▸ Sampling plan (ComboBox above the fields) | done | `ui/dock/model.rs:250`, `ui/sampling_presets.rs:35`, `ui/sampling_presets.rs:99`, `ui/search.rs:133` |
| 182 | M6 (user request 2026-10-15, "bonds that aren't true"): bond strength — the line's width and opacity follow log ρ_b (0.01 → 0.3 e/Bohr³, a display scale), its pattern the QTAIM kind (sign of ∇²ρ_b against 3× its estimated noise): solid shared, dashed closed-shell, dotted indeterminate; ring bonds keep violet but no longer dash (row 169's dashes now mean closed-shell); each bonded pair's line in the Selection tab adds "Laplacian … e/Bohr⁵: covalent (shared)" / "weak / stretched contact (closed-shell)" / "not resolved on this grid", its hover the explanation and "LSDA overbinds stretched contacts (delocalisation error): a weak contact here may be weaker or absent in reality" (also shown as a note whenever a bond is listed — review r1); View ▸ Bonds and molecules tooltip explains the styles | Overlay (3D lines); Dock ▸ Selection ▸ Bonds; Menu ▸ View ▸ Bonds and molecules (tooltip) | done | `overlay.rs:56`, `overlay.rs:62`, `overlay.rs:73`, `ui/mod.rs:800`, `ui/mod.rs:803`, `ui/mod.rs:820`, `ui/dock/selection.rs:125`, `ui/menus.rs:563` |
| 183 | Orbital viewer (Task 7, spec §3): **show** — Density (the default, today's view) / Orbitals / All orbitals coloured / Spin density, each button's tooltip carrying the facts of spec §2 that apply to it; Ctrl K entries of the same names | Menu ▸ View ▸ Density (selector, probed `menu-show-*`) and the top of Dock ▸ Orbitals (the same selector, `show-*`); anything but Density opens the Orbitals tab and asks the worker for the view (`Command::OrbitalView`) | done | `ui/menus.rs` `view_menu` ("show"), `ui/dock/orbitals.rs` `show_selector`, `ui/actions.rs` `Action::Show`, `ui/search.rs` ("Density", "Orbitals", "All orbitals coloured", "Spin density") |
| 184 | The orbital list: per spin channel in energy order, label (`2p_x` for a lone atom's labelled shells, `#k` otherwise), spin, ε in Ha and eV (through `units`, rule 3), occupation, swatch (the orbital's colour slot), HOMO / LUMO (read off the eigenvalues, `eps_range`, never a rotated member's Rayleigh quotient) with the Kohn–Sham gap line, the tags (degenerate ×d with spread, rotated for display with the set's eigensolve residual, split by the grid by s mHa, state of the box — worded by the sign of ε, not converged, smeared nucleus), "↓: none occupied", the lone atom's inverted-shell / why-`#k` lines; clicking a row picks that orbital by track | Dock ▸ Orbitals | done | `ui/dock/orbitals.rs` `list`, `row`, `RowView`, `Levels`, `box_tag`, `grid_split_tag`, `lone_atom_line`; `ui/actions.rs` `Action::PickOrbital` |
| 185 | **show unoccupied** (m per spin channel, 0–8, default 4): sent when the slider is let go; the lines "computing unoccupied orbitals…", "m requested, k shown: a degenerate set cut by the block edge is hidden", "interrupted — recomputed when idle", "showing the m = held set; requested requested", and the provider's notes; a freshly switched-on view asks for none until its first payload is in | Dock ▸ Orbitals | done | `ui/dock/orbitals.rs` `controls` ("d-orb-unocc"), `unoccupied_lines`; `ui/actions.rs` `Action::Unoccupied`; `viewer.rs` `drive_orbital_view` |
| 186 | Orbitals mode: **draw as** phase ± / \|ψ\|², and the iso slider of the ± surface (phase and spin, × the field's scale; enabled with the isosurface render mode) | Dock ▸ Orbitals | done | `ui/dock/orbitals.rs` `controls` ("d-orb-style", "d-orb-iso"), `ui/actions.rs` `Action::OrbitalPhase`, `gfx.rs` `Painter::set_view` |
| 187 | All orbitals coloured: **blend** additive / weighted (a render setting, no request), **scale** each to its peak / physical, an **include** box per orbital (default every occupied one; a unoccupied row can be added) and "include every occupied orbital" | Dock ▸ Orbitals | done | `ui/dock/orbitals.rs` `controls` ("d-orb-colour"), `row`; `ui/actions.rs` `Action::ColourBlend`, `ColourScale`, `Include`, `IncludeAllOccupied` |
| 188 | **Spin readout** (Spin density mode): n↑, n↓, ∫m = n↑ − n↓, ∫\|m\|, max\|m\|/max ρ, "no net spin" for a closed shell (the display floor), the broken-symmetry / core-polarisation note | Dock ▸ Orbitals | done | `ui/dock/orbitals.rs` `spin_lines`, `show` ("d-orb-spin") |
| 189 | **Legend chip** over the view: the phase pair (+ red, − blue, "only the relative sign of its lobes means anything"), the spin pair (↑ amber, ↓ teal), or the palette slots of the included orbitals with the scale in use | Overlay (bottom left, above the view switch) | done | `ui/overlays.rs` `legend_of`, `legend` |
| 190 | The text over the view in an orbital mode: "orbitals from t = … fs (last DFT solve). The estimated solver has no electrons" (faded layer), "needs a DFT step", "no electrons — nothing to show", the CPU teacher's "no background solver, so no orbitals: switch the solver to GPU, or to DFT", "an earlier geometry" when paused off the payload's geometry | Overlay (bottom centre, in place of the density label) and the tab's header | done | `viewer.rs` `OrbitalShown::label`, `ui/overlays.rs` `show` (label), `ui/dock/orbitals.rs` `show` |
| 191 | **colour dots by** density / orbital (dots in All orbitals coloured); dots of an orbital are ψ̃² and of the spin density \|m\| coloured by sign | Menu ▸ View ▸ Density | done | `ui/menus.rs` `view_menu`, `gfx.rs` `Painter::set_view` (`DotsLook`) |
| 192 | The orbital view's script and screenshot forms: replay words `show`, `orbital`, `orbital-phase`, `unoccupied`, `orbital-include`, `orbital-include-reset`, `colour-blend`, `colour-scale`; `--ui` keys `show=`, `orbital=`, `unocc=`, `phase=`, `blend=`, `scale=`, `dotcolour=`, `tab=orbitals`; `run_settings` carries the view across a solver switch | — | done | `ui/actions.rs` `describe`, `parse`; `screenshot.rs` `ShotUi`; `ui/mod.rs` `run_settings` |
| 193 | Activity (spec §3.1): the **status chip's state text is the activity headline** while work runs — "Sample 42 of 273 · bond scan · 1:40 left · iteration 9 / 200" (the references stage names the last reference that finished, from this pass's rows: "reference (last done: H)"), "Refitting · 250 samples", "Solving · iteration 7 / 200 (last solve 11)", "DFT fallback · …", "Solving (CPU reference) · 12 s", "Relaxing (FIRE) · step 31" — with a pill for a background audit and one for the orbital viewer's eigensolve; idle states keep their old texts; rebuilt at most every 100 ms | Overlay status chip | done | `ui/overlays.rs` `status_chip`, `ui/activity/headline.rs` `headline`, `display` |
| 194 | Activity (spec §3.2, ruling A1): a 3 px **progress bar on the toolbar's top edge** — filled for a pass and bond references, a sweeping segment otherwise, an amber tail while an audit runs; hover shows the headline, a click opens the Activity tab | Toolbar top edge | done | `ui/activity/mod.rs` `bar`, `ui/toolbar.rs` `show` (returns its rect) |
| 195 | Activity tab: **Jobs** — one row per running or waiting piece of work with its progress, elapsed / ETA and the one existing control (Cancel, Stop, Pause, Start anyway; "can't be cancelled" says why on hover) | Dock ▸ Activity | done | `ui/activity/jobs.rs` `rows`, `show` |
| 196 | Activity tab: **Counters** — samples in the model, teacher solves for this model (with the split; "≥ n (earlier solves unrecorded)" for a loaded file), teaching solves this session | Dock ▸ Activity | done | `ui/activity/log.rs` `counters`, `counter_values` |
| 197 | Activity tab: the **solve log** — newest first, 5 000 rows, filter chips, run dividers, the rule and outcome in plain words with the technical text on hover | Dock ▸ Activity | done | `ui/activity/log.rs` `show_log`, `ui/activity/words.rs` |
| 198 | Activity tab: the **trust checklist** (a drawn check or cross per condition, the number against its limit, "this is why" on the first unmet one, the box line first when the box blocks trust, the audit line worded from the last audit and the fit stamp, class lines, the stuck variant); the Model tab's **trusted** line and the toolbar's trust tooltip use the first unmet line with the fit's own words on hover | Dock ▸ Activity, Dock ▸ Model, Toolbar ▸ Solver | done | `ui/activity/trust.rs` `lines`, `show`, `ui/mod.rs` `trust`, `ui/dock/model.rs` `readouts`, `ui/toolbar.rs` `solver_sub` |
| 199 | Activity tab: **Training** (opened on demand: the worker builds the views only then) — rmse per refit, model vs DFT (E \| F), one curve per species pair with ±1σ band, the fit's residual scatter, refused stretches dashed and labelled, the uncertainty lane, pair-distance and burst-temperature histograms, the age line | Dock ▸ Activity | done | `ui/activity/plots.rs` `training`, `rmse`, `parity`, `pair_curve` |
| 200 | Activity tab: the **timeline** (fallbacks, audits, refits, quarantines, passes over session time) and the measured bookkeeping footer | Dock ▸ Activity | done | `ui/activity/plots.rs` `timeline`, `ui/activity/mod.rs` `overhead_footer` |
| 201 | The Activity tab's script and screenshot forms: `--ui` keys `tab=activity`, `wait=model`, `training=open`, `plan=quick\|gentle\|standard\|hot`; replay words `expect solve-log N`, `expect counters` | — | done | `screenshot.rs` `ShotUi`; `replay.rs` `Expect` |
| 202 | M7 Task 9 (spec §4.7): **Physics ▸ Ionic potential** — all-electron (σ = h) / cores only / all species (`theme::segmented` over `IonicPolicy::ALL`, tooltip `IONIC_TIP`), the note that a change replaces the scene, the honesty line "E_KS is a valence energy under a pseudopotential (spec §6.5)" and, for each pseudised species of the last solve, "O  Z_v 6 · r_c 1.35 Bohr · … · energies certified at h ≤ 0.20 · egg-box certified at h ≤ 0.15" (min(h_eb, h_req); F: none on the ladder) and the amber line "egg-box force not certified at this grid" (text only, never a refusal; its hover is the measured O₂ record). Ctrl K "Ionic potential"; script word `ionic none\|cores\|all`; `--ionic` | Menu ▸ Physics (anchor `m-ionic`) | done | `ui/menus.rs` `ionic_controls`, `ui/pseudo.rs`, `ui/actions.rs` `Action::Ionic`, `ui/search.rs`, `main.rs` `--ionic` |
| 203 | M7 Task 9: **View ▸ Density ▸ with partial core** — adds ρ_pc (the generator's core model, not solved) to the density shown; applies from the next step; Ctrl K "Partial core density"; script word `partial-core true\|false` | Menu ▸ View ▸ Density (anchor `m-density`, probe `show-partial-core`) | done | `ui/menus.rs` `view_menu`, `ui/actions.rs` `Action::ShowPartialCore`, `Command::DisplayPartialCore`, `GpuSolver::copy_density_with_core_to_volume` |
| 204 | M7 Task 9: Scene ▸ Quality marks a level whose h exceeds a pseudised species' h_req "needs h ≤ 0.20 (O)" (from the last solve's record, else the cache — never generating), on the named levels and the Custom slider | Menu ▸ Scene ▸ Quality | done | `ui/menus.rs` `quality_controls`, `ui/pseudo.rs` `needs_finer` |
| 205 | M7 Task 9: Readouts — the energy row reads "E_KS (valence, pseudopotential)" under a pseudopotential, the open-shell occupation line of Task 6 ruling A.6 ("open shell: 1 e shared equally over 3 degenerate levels (↓), the spherical-atom ensemble; LSDA's symmetry-broken integer state lies ΔE lower"), the core-overlap and occupation-change warnings in Health | Dock ▸ Readouts | done | `ui/dock/readouts.rs`, `ui/pseudo.rs` |
| 206 | M7 Task 9: a pseudised nucleus — the badge ends "· pseudo, Z_v = 6, r_c = 1.35 Bohr, needs h ≤ 0.20"; the atom card says "pseudo-ion in the solve: 6 valence electrons; the 2 core electrons are frozen into the potential"; Dock ▸ Selection has the collapsing "pseudopotential (generation record)" | Overlay; Dock ▸ Selection | done | `viewer.rs` `badge`, `atom_card.rs` `pseudo_line`, `ui/dock/selection.rs` `pseudopotential` |
| 207 | M7 Task 9: core overlap — each overlapping pair red between the nuclei (`overlay::core_overlap_lines`) and a toast with the warning (once per pair); the orbital list shows fractional occupations (f = 1/3) and quotes no HOMO–LUMO gap for a partly filled shell | Overlay (3D lines, toast); Dock ▸ Orbitals | done | `overlay.rs`, `viewer.rs` `overlay_lines`, `ui/logbook.rs`, `ui/dock/orbitals.rs` `Levels` |
| 208 | Show what training is solving (user request 2026-10-18): **View ▸ Display aids ▸ show what training is solving** (default on; display only — writes `UiState::show_ghost`, no command, not recorded, so a `--replay … --screenshot` renders it at its default). While a sampling pass runs the 3D view draws the job's ghost (`ghost.rs`, from `DynSnapshot::ghost` = `sim::PassGhost`) instead of the scene — reference atom, nudge with arrows, scanned pair highlighted with its live `O–H · 1.50 × r₀ · 2.71 Bohr`, hot burst with a trail — with the cell and walls kept, the scene's spheres, bonds, labels and cloud hidden, and a green / red / grey flash (0.5 s) per finished sample; the scene returns when the pass ends, is cancelled or fails. Ctrl K "Show what training is solving"; `--ui ghost=on\|off;wait=nudge\|scan\|burst` | Menu ▸ View ▸ Display aids | done | `ghost.rs`, `viewer.rs` `ghost_now` / `drawn_nuclei` / `overlay_lines` / `label_overlay`, `ui/menus.rs` `show-ghost`, `ui/search.rs`, `screenshot.rs` `ShotUi::ghost`, `sim::Trainable::pass_ghost`, `providers::sampling::Sampler::ghost` |
| 209 | **Trails** (user request 2026-10-17): a checkbox in Display aids; on, it shows **length** (5 fs – 10 ps of simulated time, logarithmic slider, default 100 fs, typed values like `250` or `2 ps` parsed) and **opacity** (0–100 %, default 70 %). A single thin line per atom in its element colour, alpha fading linearly from the opacity at the newest end to 0 at age = length, through `render::lines`, depth-tested like the bonds. Off: no history collected; on starts from now; a teleport clears that atom's trail, a nucleus-count change / scene replace / relax / backend switch / time going backwards clears all, a drag keeps it; at most 4096 points per atom (the oldest half is thinned). Display only (`UiState`, not recorded); Ctrl K "Trails"; `--ui trails=on\|off`, `trail_fs=<fs>`, `trail_opacity=<0-100>` | Menu ▸ View ▸ Display aids | done | `trails.rs`, `ui/menus.rs` `trail_controls`, `ui/search.rs`, `viewer.rs` `record_trails` / `overlay_lines`, `screenshot.rs` `ShotUi` |
| 210 | M7 Task 13 (user request 2026-10-15, "display the bond angles on the bond chains"): **bond angles** — an arc and a degree label ("104.4°") between each pair of drawn bonds at an atom, computed from the nuclear positions drawn now (atan2(\|u×w\|, u·w)); the arc is a 4-strand bundle on a tube of 0.0225 Bohr (half the bond radius, fixed width) and its radius is at least 0.45 Bohr, 1.3 × the vertex sphere's drawn radius where that is larger (so it clears the sphere; scale from the view mode), at most 0.6 × the shorter bond; an atom with at most 3 bonds shows all its angles, one with more only while selected or hovered (Select tool); the arc takes the weaker bond's kind (solid shared, dashed and faint closed-shell, dotted indeterminate, violet dashed ring) and fades with the stale bonds; Dock ▸ Selection lists the selected atom's angles ("∠H–O–H 104.4°") with the surface note and appends "(valence-density kind)" + the r_c explanation to a bond whose bond point lies inside a pseudised atom's r_c; View toggle on by default, tooltip, Ctrl K "Bond angles", script word `angles true\|false`, `--ui angles=on\|off` | Menu ▸ View ▸ Bonds and molecules ("bond angles", probed `show-angles`); overlay (3D lines + labels); Dock ▸ Selection ▸ Bonds | done (the Selection tab's pseudised species are `pseudo::pseudised`: the valence-density clause and the surface note are live; the view's arc and label sphere scale follows the painter's own layer test) | `overlay.rs` `bond_angles`, `shown_angles`, `angle_style`, `angle_arc_lines`, `angle_label_anchor`; `viewer.rs` `angle_marks`, `overlay_lines`, `label_overlay`; `ui/menus.rs` `ANGLES_TIP`; `ui/actions.rs` `Action::ShowAngles`; `ui/search.rs` ("Bond angles"); `ui/dock/selection.rs` `bonds`; `gfx.rs` `nucleus_scale`, `mode_of_request`; `ui/mod.rs` `angle_surface_note`, `valence_density_kind`, `VALENCE_KIND_NOTE`; `screenshot.rs` `angles` |

(Rows 202–207 are M7 Task 9's: Physics ▸ Ionic potential, View ▸ Density ▸ with partial core, the Quality marks, the Readouts' valence and occupation lines, the pseudo badge / card / record, and the core-overlap tint.)

## Deviations from the mockup's wording (coordinator rulings, fix round 1)

- **Rows 5 and 6, refusals and status lines (review I2).** The mockup moves them to toasts plus the Log; the old panel kept them in place. Both are in place again: the status chip shows the status line until it is replaced or cleared (amber when it asks for something, e.g. "paused: … press Play to continue"; Play clears it) and the driver's refusal until the next accepted command clears it; the Log tab also has a last refusal line. A repeated message gets fresh feedback: UiState::set_status bumps status_seq and a refusal is keyed on (text, Snapshot::edits), so Save twice or Space twice during a pass re-toast and re-log. Refusal toasts sit above the toolbar (bottom centre), status toasts at the bottom right.
- **Row 194, the progress bar (activity ruling A1).** The mockup has no bar. The toolbar is at the bottom of the window, so "under the toolbar" would be the least-watched edge: the bar sits on the toolbar's **top** edge, the seam with the 3D view, next to the controls.
- **Activity wording deviations (final review M8).** The log's time hover says "x s into the session" (the events carry a monotonic instant, not a wall-clock time); the spec asked for the wall time of the event. The strip's peek shows the headline, the three counters and the newest log row.
- **Rows 193–201 are numbered 193+ because 183–192 were taken by the orbital viewer (merged first).**
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
