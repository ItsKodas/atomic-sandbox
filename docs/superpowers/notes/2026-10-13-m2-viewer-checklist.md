# M2 viewer — manual checklist status

The viewer's manual checklist is Task 7, Step 5 of `docs/superpowers/plans/2026-10-13-m2-viewer.md`.
This note records how far each item was verified when M2 landed. Since M6 Task 8 the items
that need a mouse run as **replay scripts** (`crates/app/replays/*.replay`, headless, every
`cargo test -p app --release` via `replays_pass`; "Replay `name`" below, mapping in the last
section). Re-run what is still manual by hand after UI or render changes: `cargo run -p app --release`.

"In-window" means it was seen in a running window: a capture or the stderr log, from the
implementer's own instance or the user's live session. "Code reading" means the logic was
checked (and unit-tested where noted), but nobody exercised it in a window.

| Item | Status | Evidence |
|---|---|---|
| Window opens (RX 7900 XTX, Vulkan, `Rgba8UnormSrgb` surface) | In-window | log + window capture |
| H₂ appears with its density; snapshot uploads with right dims | In-window | 49³ × 0.25 Bohr, ρmax 0.2231, E −0.962716 Ha, 8 SCF steps, ~5.9 s; capture shows density + spheres |
| Resize | In-window (user) | maximised to 2560×1369; correct rendering afterwards, no panic |
| Resize / Outdated while the worker submits on the shared device | Headless test + code reading | Seen once (2026-10-14): wgpu 30 panicked ~2 s after launch with "Failed to wait for GPU to come idle before reconfiguring the Surface" (`ConfigureSurfaceError::GpuWaitTimeout`, a validation error wgpu-core raises when another thread submitted during `configure`'s idle wait). `SurfaceState::resize` now captures it in a thread-local error scope and returns `Err` leaving the old surface; `Gfx::resize` logs "surface: reconfigure … deferred to the next frame" and `redraw` retries while `config` differs from the window size. `crates/app/tests/surface_reconfigure.rs` recreates the race (a worker thread submitting with readback waits, 100 reconfigures of a hidden window, ~3 s): without the scope it panics at the reported wgpu line; with it 13–25 % are refused and the retry succeeds. Manual check: launch, press Play, resize/maximise repeatedly for ~10 s: at most the deferred line on stderr, never a panic |
| Minimise | Code reading; no script (window surface) | `SurfaceState::resize` ignores 0 sizes; `redraw` returns early at 0×0; `about_to_wait` stops requesting redraws while minimised (polls instead) |
| Preset switching (He, H₂O) | In-window (user) | He → E −2.138685; H₂O → spheres update at once, spinner shown, old density kept meanwhile |
| Isosurface mode | In-window (user) | capture |
| Solver error → red message, old picture stays | In-window (user) | charge > protons on H₂ → red "exceeds the 2 protons" line, previous density still drawn |
| Gamma slider → volume updates | In-window (user) | capture at gamma 0.55 |
| Orbit / pan / zoom | Replay + code reading | Replay `camera_leaves_scene_alone` (middle-drag, right-drag, wheel, a Select left-drag on empty space: nothing sent, nothing selected). The camera's maths: `render::Camera::orbit/pan/zoom` unit tests; `Viewer::zoom` floors the distance at 2.0. How it *feels* in a window: manual |
| Place adds a nucleus on the focus plane, density updates | Replay | Replay `place_three_build_water` (O and H placed by clicks land within 2e-4 Bohr of the focus-plane points aimed at; the solve and the bonds follow), `pin_holds`; `tools::place_point` unit-tested |
| Display aids: cell box, floor grid (xz, +y up) with axes, placement-plane grid in Place mode, 1 Bohr / 1 Å spacing | Code reading + headless | `render::lines` pass tested headless (lit pixels, occluded by spheres, no depth write); `overlay::build` unit-tested |
| Distance guides: faint lines + Bohr/Å labels from a dragged nucleus (or the Place cursor) to nuclei within 6 Bohr | Code reading; no script (drawn overlay — replays assert on snapshots, not pixels) | `tools::distance_guides` / `tools::project` unit-tested; overlay in `Viewer::distance_overlay` (egui foreground layer) |
| Select + drag, Delete key | Replay | Replays `select_drag_paused_teleport`, `delete_and_space`; `tools::pick` / `tools::drag_point` unit-tested; drag sends `MoveNucleus` per motion event, the worker coalesces (`sim::queued_commands_are_coalesced_into_one_solve`) |
| Coincident placement → red "coincide" error | Code reading; no script (a script has no `expect error` yet) | worker path: `sim::provider_failure_pauses_with_message` (M4; was `coincident_nuclei_report_an_error_not_a_crash`); UI path is the same as the charge errors above |
| Apply grid 33/0.3 recreates the volume | Code reading; no script (Apply grid sends the panel's pending fields, which a script does not carry: `Action::describe` is `None`) | upload recreates on `volume_dims != [n;3]` (also hit by H₂ 49³ → H₂O 57³); worker: `sim::grid_change_resizes_the_density` |
| Close → window disappears at once, process exits | In-window (scripted) | `WM_CLOSE` during an H₂O solve → exit in ~140 ms, exit code 0 (commit 90d5a76) |

Known gap: the H₂O interactive solve at 57³ takes ~2 min (116 s, 13 SCF steps), not the plan's
"~10 s". This is solver performance, not the viewer; the UI stays responsive because the solve
runs on the worker thread.

## M4 Task 6 — dynamics in the viewer

Checked on 2026-10-13 against `m4-dynamics` (RX 7900 XTX, 240 Hz, GPU backend unless stated),
with stderr redirected to a log: the viewer prints one line per second while playing (time, T,
E_total, E_conserved, drift, fs/s, ms/step, SCF steps, the 0–1 distance), a frame-interval line
every 5 s while playing, and a line for every snapshot that leaves the worker idle. The window was
driven by posted Space keys and `WM_CLOSE`. Posted mouse input does not reach egui (the real cursor
was outside the window), so panel clicks were not scripted. The user was also using the window
during two sessions; the items they exercised are marked "In-window (user)", with the evidence
taken from that session's log. "Headless" means a viewer test that drives the same `Viewer`
code the window runs: `cargo test -p app --release`.

| Item | Status | Evidence |
|---|---|---|
| Starts on H₂ at R = 1.6 Bohr, paused | In-window | log: E_KS −0.980640 Ha, 8 SCF steps, 196–250 ms cold (GPU), 6.4 s (CPU). The capture shows "paused — converged", the Play button and the empty sparkline placeholder |
| Play / Pause with Space | In-window + replay `delete_and_space` | the play log starts at once; Pause logs an idle snapshot within one step. Once, a posted Space did not take and the next one did |
| Vibration | In-window | d(0,1) runs 1.600 ↔ 2.017 Bohr. Minima at t = 0, 20.75 and 52.0 fs give a period of **10.4 fs**. T runs 0 ↔ 890 K. E_total stays in −0.980640 … −0.980669 Ha, a Verlet swing of 3e-5 Ha (≈ (ω dt)²/4 of the 4e-3 Ha exchanged). After 100 steps the drift is +8e-5 … −7e-7 Ha/ps, ~1e-5 typical |
| Speed of the run | In-window | 2.2–3.5 fs/s at 77–120 ms per step (4–7 SCF steps, ~240–280 readbacks), 49³ |
| Frame time while playing | In-window | interval p50 **4.2 ms**, p95 5.1–5.8 ms, max 6–38 ms per 5 s window. The 30–38 ms maxima fall in the first window after Play |
| Readouts and sparkline | In-window | captures show T, KE, E_KS, E total, E conserved (Ha and eV), drift, SCF steps / readbacks / ms per step, fs/s and steps per tick, and the E_total/T sparkline in anti-phase. The drift turned red in the runaway below. Amber was seen in the log only (+1.07e-4 Ha/ps), not in a capture |
| Play / Select / orbit with the mouse | In-window (user) | the user's session: Play, the Select tool, a camera orbit and a zoom |
| Drag while playing = spring | In-window (user) + test + replay `select_drag_playing_spring` | user log: E_conserved separates from E_total by the spring energy while the drag is held, and T spikes to 2.4e4 K. `drag_while_playing_uses_the_spring_not_teleport`: max \|Δr\| per snapshot 0.021 Bohr, no jump |
| Select + drag while paused = teleport | In-window (user) + replay `select_drag_paused_teleport` | user log: an idle snapshot after the move, d(0,1) 4.19 → 2.45 Bohr, 11 SCF steps |
| Provider failure pauses with a message | In-window + test | GPU: "SCF did not converge after 200 iterations (Δρ = 1.9e-4)" at d = 2.61 Bohr during the user's run paused it with the red line; Space resumed. `provider_failure_pauses_with_message`: "nuclei 0 and 1 coincide" on both backends |
| CPU backend playing | In-window (`--backend cpu`) | 2.4–3.6 s per step, 0.06–0.11 fs/s. The trajectory matches the GPU run at step 21 (d(0,1) 2.0167 Bohr, T 0.2 K) and step 25 (1.9838 Bohr, 220.8 K). Frame interval p95 4.4–4.9 ms during CPU steps |
| Close during a CPU MD step | In-window (scripted `WM_CLOSE`) | the process exited in 142 ms |
| CPU ↔ GPU radio | Headless | `switching_backend_respawns_the_worker_with_the_current_scene` (the panel's settings are re-sent; the run restarts paused) |
| Pin, Rewind, thermostat, gravity preset | Headless + replay `pin_holds` (pin) | `panel_controls_reach_the_driver`: the pinned nucleus is bit-still over 5 steps; Rewind flips every velocity exactly; switching the thermostat on at rest gives T = 300 K and the Maxwell–Boltzmann note; g = 1.0836e-22 Bohr/a.u.² reaches the driver |
| Place / Delete while playing | Headless + code | `mirror_follows_the_dynamics_and_keeps_unconfirmed_edits` (an addition while playing shows at once and survives stale snapshots); driver: `add_and_remove_while_playing` |
| Drag target marker, pin squares, box-face warning | Code + unit test | `Viewer::distance_overlay`; `box_warning_flags_nuclei_near_or_beyond_the_face` |
| Save / Load with velocities | Unit test | `converts_to_and_from_the_dynamics_scene_with_velocities` (vx vy vz written only when non-zero) |
| Display aids during play | In-window | the captures show the cell box, the floor grid and the axes. The distance guides were not exercised (they need a mouse drag) |

**Runaway seen in the user's session (not a viewer bug; reported to the controller).** A drag
while playing gave the molecule centre-of-mass momentum, and with COM correction on nothing
stopped it at the box face. E_total rose from −0.934 to +0.16 Ha while E_conserved stayed flat at
−0.932: the energy entered through the COM correction's booked work. T reached 1.2e5 K, and the
nuclei left the 12 Bohr box (d(0,1) up to 35 Bohr; SCF steps up to 137). The panel now warns when
a nucleus is within 1.5 Bohr of the box face, or outside the box.

### Task 6 fix round 1 (R12, R13, review I1/I2/m1)

None of the items below was exercised in a window. They need a mouse, and posted mouse input
does not reach egui. Since M6 Task 8 each row names its replay script, or why it has none.

| Item | Status | Evidence |
|---|---|---|
| A nucleus closer than 3h to a face, or outside the cell, pauses with "left the simulation cell" (R12) | Unit + worker test; no script (the worker test is the check; a replay would need an `expect error`) | provider `nuclei_near_or_outside_the_cell_face_are_refused` (CPU and GPU, refused before any solve); app `a_nucleus_at_the_cell_face_pauses_with_a_message` |
| Drag spring force capped at 0.1 Ha/Bohr, work booked exactly (R12) | Unit test + replay `select_drag_playing_spring` (no jump to the cursor through the input path) | `drag_force_is_capped_and_its_work_booked`: the first step is exactly F_max; E_total peaks at the capped potential, 0.18994 vs 0.19000 Ha (uncapped would be 1 Ha); \|ΔE_conserved\| ≤ 1.3e-3 Ha |
| "work done by tools" readout (E_total − E_conserved) | Code; no script (panel layout: a readout's text) | `ui::dynamics_readouts`, with a tooltip on the drag-speed trade-off |
| A paused teleport warm-starts (R13) | Worker test; no script (the worker test is the check) | `moving_a_nucleus_warm_starts` (h2 49³, 0.02 Bohr): warm 381 readbacks / 7 steps vs cold 477 / 8, ratio 0.80 < 0.85; ΔE 1e-8 Ha |
| The drag mode follows Play/Pause at every motion event, never a teleport while playing (I1) | Unit test + replays `select_drag_paused_teleport`, `select_drag_playing_spring` | `tools::drag_mode_follows_the_play_state_at_each_motion`. A select-click without motion no longer tethers the nucleus (review m2) |
| Drag targets, teleports and placements clamped into the cell minus 3h and ½h (I2) | Unit test; no script (the unit test is the check) | `tools::clamp_keeps_positions_inside_the_validation_margin`. The marker draws the clamped target |
| Focus loss or the cursor leaving the window while a button is held counts as a release (m1) | Unit test | `replay::tests::viewer_input_is_the_only_path` (`Event::FocusLost` / `CursorLeft` mid-drag end it; `from_winit` maps `Focused(false)` and `CursorLeft`); now `Viewer::release_all` |
| Atom card (M4+, user request): selecting a nucleus shows a top-right Bohr-style schematic with Z protons, N neutrons (most abundant isotope), electron shells, and the not-to-scale / shared-electrons caveat | Code reading + unit tests (`atom_card::tests`); no script (panel layout) | `crates/app/src/atom_card.rs`, hooked in `Viewer::run_ui` |

## M5 Task 8 — the estimated solver in the viewer

Checked on 2026-10-14 against `m5-surrogate` (RX 7900 XTX, 240 Hz), the window launched from
`target-shot` with `--estimated [--speed 1000] [--audit n]`, stderr to a log (the viewer now
prints one line per second while sampling, and the play line names the source, the fallback
fraction and the model's samples/epoch). As in M4 the window was driven by posted Space keys and
`WM_CLOSE` only and captured with `PrintWindow`; posted mouse input does not reach egui. In one
session the user drove the window by hand (marked "In-window (user)"). "Worker test" / "Viewer
test" = `cargo test -p app --release`.

| Item | Status | Evidence |
|---|---|---|
| Estimated with no model starts a sampling pass; Play is disabled while it runs; progress bar, stage, last energy, skipped/failed | In-window + worker test | H₂ 49³, default plan: 273/273 jobs in ≈ 33 s (four sessions: 32–35 progress lines at 1/s), 0 skipped, 0 failed, trusted at the end (273 samples, Q′ 29). The capture shows "sampling the teacher: hot burst 128/273", the bar and the paused dynamics. `enabling_estimated_without_a_model_starts_a_sampling_pass` (41³, short plan: 83 jobs in 10 s, Play refused with the driver's message, progress monotone) |
| The pass does not unpause anything | In-window | session 3: idle snapshot after the pass, nothing played until the posted Space |
| Play after the pass: surrogate answers, fs/s | In-window | speed 1000: **≈ 300 fs/s** at audit 200 (the original default; 410 headless), **≈ 3300 fs/s** at 2000 (the default since fix round 1, ruling S14), **≈ 30 000 fs/s (1.2e5 steps/s)** with `--audit 0`. Worker test (fix round 1: Δsim/Δwall over 3 s, not a per-tick median): 200/200 surrogate steps, 3 474 fs/s at the default vs 3.12 fs/s GPU DFT (×1 112); audits off 31 655 fs/s (×10 135) — the run of record in the Task 8 report (other runs 2 827–4 065 fs/s with the GPU shared). **Task 9** (background audits, variance cadence 4): 38 816 fs/s at the default in the same worker test, 39 841–41 969 fs/s headless at 49³ — in-window not re-measured |
| Fix round 2: a command flood while playing paced | Worker test | `a_command_flood_neither_speeds_up_nor_stalls_a_paced_run` (analytic Morse H₂, speed 1, a DragMove every ~4 ms for 2 s): sim rate 15.5 fs/s vs paced 15.6 (60.9 fs/s with commands running full ticks, as before the fix), next step 23.7 ms after the last command |
| Frame time while the surrogate plays | In-window | interval p50 4.2 ms, p95 4.7–6.0 ms, max 5.5–13.7 ms per 5 s window, at every audit setting |
| Readouts: source badge, uncertainty vs threshold, trusted, samples, epoch, last fit, last refit, fallbacks, teacher gap, mismatch work | In-window (captures) | e.g. "surrogate", 3.9e-5 / 4.9e-3 Ha/Bohr, trusted yes, 350 samples, epoch 77, rmse 1.2e-5 Ha/atom / 3.9e-5 Ha/Bohr, refit 22 ms; drift ±4e-7 Ha/ps |
| Untrusted reason shown | Worker test (by reading) + capture | "no model yet" before the pass; at 33³ the pass ends untrusted with "max training energy residual 2.7e-3 > 1.0e-3 Ha/atom" (measured while writing the tests: why they run at 41³) |
| Sparkline coloured by source, fallback dots, legend | In-window (capture) + headless screenshot | green E-total line with amber dots at the audits; `docs/images/surrogate_h2.png` |
| Last DFT density faded to 30 % with "density from the last DFT step (t = … fs)" | In-window (capture) + viewer test + render test | label "density from the last DFT step (t = 3831.00 fs) — the surrogate has no electrons"; `estimated_radio_samples_fades_…` (fade 0.3, label), `render::volume::fade_scales_the_output_in_both_modes` (0.3 × the centre pixel ± 1.5/255 in both modes) |
| Place an atom while the surrogate plays; dt change | In-window (user) | the user's session: a third H placed while playing with dt 0.010 fs — the model learned H₃ from fallbacks (273 → 404 samples, epoch 131) and kept answering. Its refit jumps went into "work done by tools" (−2.19 Ha, E conserved continuous), now explained in that tooltip |
| Drag far → fallback lights up, the model learns | Worker test (not in a window: needs a mouse) | `fallbacks_are_visible_in_snapshots`: teleport to R = 0.8 (Task 9: R = 4 in the 41³ box is box-affected now and not trained on) → Fallback by the envelope (4.4 training spreads), uncertainty 3.8e-3 vs bound 1.0e-2, teacher gap +4.58e-3 Ha, 83 → 84 samples, refit 5 ms; the next evaluation at R = 0.8 is Surrogate (uncertainty 3.9e-4) |
| Pause / Cancel during a pass | Worker test | `pause_and_cancel_abort_a_sampling_solve_promptly` (H₂O 97³; since Task 9 the automatic pass is first an offer, "Start anyway" starts it): Pause → snapshot in 169 ms, the pass continues; Cancel → cancelled pass in 17 ms (earlier runs 20 / 39 ms) |
| Save / Load model, `<scene>.model`, provenance refusal | Worker test | `model_sidecar_saves_and_loads_with_the_scene`: saved, a new worker restores 83 samples and the identical fit, first step Surrogate (0.05 ms); a 33³ worker refuses the 41³ file and starts no pass |
| estimated → GPU → estimated keeps the model | Viewer test | `estimated_radio_…_keeps_the_model_across_a_switch`: the session autosave restores 84 samples and the fit, no new pass |
| Forget model (two clicks within 4 s), Sample more, audit field, sampling-plan fields, pin / thermostat under the surrogate | Code reading only; no script (panel layout: Sample and the plan fields carry panel state a script does not; `action forget` ×2, `action audit N`, `action pin`, `action thermostat` are replayable but not yet scripted under the surrogate) | re-check by hand. Pins and the thermostat are driver features independent of the provider (M4 tests). The first pass starts at once when estimated is picked — the panel's plan if it fits about 2 minutes, else scaled down, else an offer with "Start anyway" (Task 9); edit the plan by Cancel → change → Start |
| Fix round 1: smooth motion (nuclei interpolated between the last two playing snapshots) | Unit test + in-window (frame rate only) | `interpolation_is_monotone_and_snaps_on_count_change`. In the window the frames stayed at p50 4.2 ms / p95 5.9–6.4 ms with it on; whether the motion *looks* smooth was not judged by eye from the log or the stills — no script (a judgement by eye) — re-check by hand (toggle "smooth motion" in the display row) |
| Fix round 1: tick pacing and "watchable" | Worker test + unit test + in-window | speed 1 on the surrogate: 15.7 fs/s (0.25 fs per 16 ms; 8.3 before the fixed-schedule deadlines, Windows timer granularity); `the_watchable_speed_…`: 1 step/tick, 15.6 fs/s at dt 0.25 on the surrogate, 2.5 fs/s on DFT. In the window the button read "watchable (≈ 1 fs/s)" during the user's far drag (every step a 250 ms fallback) |
| Fix round 1: a far drag in the window (user) | In-window (user) + replay `surrogate_far_drag_recovers` | the user dragged H₂ apart while playing on the surrogate: fallbacks 45 % of 100, T 3600 K, 250 ms/step; the refit then **untrusted** the model ("max training energy residual 8.6e-3 > 1.0e-3 Ha/atom", 322 samples) — Task 7's "one stray sample can untrust a model", now seen in a window; the panel showed the reason |
| Fix rounds 3–4: once it falls back it comes back (the user's report: 100 % fallbacks, "3540 samples epoch 0 untrusted") | Worker + `sim` tests | root cause: one sample the model cannot fit failed the training-residual trust rule at every later refit, and every fallback kept its sample. The sample in the user's session was H₂ pulled to R = 8 Bohr in the 10 Bohr box (41³) — +0.3 Ha of **box confinement** (its nuclei 1 Bohr from the faces; fix round 4 corrected the round-3 "beyond the cutoff / restricted singlet" story). Now: a degree-cap retry, then quarantine (ruling S15), and box-affected answers are not trained on. App tests: `after_a_far_drag_the_surrogate_answers_again` (R = 8 → not kept, R = 3.3 → Surrogate again within the asserted ≤ 20 fallback steps and ≤ 2 refits; the test prints the count — 1 in the Task 8 round-4 report, 15 with 0 refits since the final fix wave (deterministic over three runs): the 41³ pass's stretched scan end is box-affected, now flagged as such and left out of the fit (final review I1), so R = 3.3 starts outside the trained envelope and the box-affected answers there are not kept — the model answers once the vibration brings the bond back in), `a_far_geometry_in_a_small_box_falls_back_and_is_not_trained_on`; `sim` tests with analytic teachers for the quarantine itself: `a_stray_sample_is_quarantined_then_superseded_and_trust_returns`, `a_quarantined_sample_is_released_when_the_model_reproduces_it`, `a_molecule_parked_beyond_the_cutoff_neither_grows_the_store_nor_untrusts_the_model`, `a_box_affected_answer_is_used_but_not_trained_on`. Not re-checked in a window; replay `surrogate_far_drag_recovers` does the hand check (drag H₂ to ≈ 3.3 Bohr while playing, hold, let go: the model answers again 400 steps later) |
| Task 9: "trusted, Fallback, 100 % of 100, teacher gap +2.9e-3 Ha" near equilibrium (the user's round-3 log) | Worker tests | root cause: box confinement (≈ 50 Ha per e/Bohr³ of face density) under the old 1e-3 box rule — an audit near a face failed on +6.4e-3 Ha of confinement, the refit quarantined that sample, and its descriptor-space fence covered the equilibrium bond everywhere in the cell. `a_molecule_parked_near_a_face_is_not_fenced_by_the_box` (H₂ 41³ parked 1.5 Bohr off-centre, audits every 100: 0/251 fallbacks, the audits "not judged (box-affected)", 0 quarantined; back at the centre Surrogate after 1 step, next audit −5.3e-5 Ha); `after_a_hot_excursion_the_model_answers_at_equilibrium_again` (12 random hot teleports, then equilibrium: Surrogate after 1 step, next audit −5.4e-5 Ha; with the old 1e-3 rule the model ended untrusted, 9 quarantined). In a window: not yet |
| Task 9: audits in the background (ruling S14) | Worker tests + headless screenshot | the audit step is the model's; a second GpuKs solves it meanwhile ("last audit: in flight on the background teacher", then "passed: −5.3e-5 Ha (274 ms DFT, judged 17 steps after dispatch)"). H₂ 41³ at audit 2000: **38 816 fs/s** (Task 8: 3 474), audits off 43 561; headless 49³ screenshot run 39 841–41 969 fs/s over 2 s. `sim`: `a_background_audit_runs_while_the_model_answers`, `a_failed_background_audit_untrusts_and_books_the_flip` (ΔE_c across the flip ≤ 1e-5). In-window fs/s: not re-measured |
| Task 9: sampling-pass ETA, 2-minute first pass, "Start anyway" | Worker + unit tests | `the_first_pass_keeps_to_its_time_target_or_asks` (H₂ default plan as is at 0.12 s a solve, scaled to 133 solves at 0.9 s, an offer at H₂O 57³ 2.4 s: 210 solves ≈ 504 s); `pause_and_cancel_…` (H₂O 97³: the automatic pass is an offer — 210 solves ≈ 800 s — and "Start anyway" starts it); the button's "N solves, ≈ T" and the progress bar's time left seen in the headless screenshot ("273 solves, ≈ 24 s"). The two-click "Start anyway" on the manual button: code reading only; no script (Sample carries the panel's plan) |
| Task 9: readouts — model / DFT step costs, "model refit" and "model force mismatch" rows, "yes, but not answering here" | Headless screenshot + `sim` tests | `a_slow_dft_step_does_not_throttle_the_model_steps_after_it` (400 steps in the tick after a 100 ms fallback; [399, 1, …] with one mixed average), `a_trusted_model_that_keeps_falling_back_says_so`; the rows in `docs/images/surrogate_h2.png` |
| Fix round 1: leaving estimated never blocks the window; the session autosave survives quick toggles and a refused autoload | Worker + viewer tests | `leaving_the_estimated_solver_neither_blocks_nor_loses_the_autosave` (switches during a cold 97³ evaluation: worst `run_ui` 5.2 ms, file unchanged), `a_worker_shut_down_before_ensure_model_leaves_the_autosave_alone`, the refused-autoload retire in `model_sidecar_saves_and_loads_with_the_scene` |
| Crash seen once at start-up (M5) | In-window, not reproduced | the first `--estimated` session panicked 2 s in, while sampling, in `Surface::configure` ("Failed to wait for GPU to come idle before reconfiguring the Surface") — the resize path of `gpu::SurfaceState::resize` while the worker kept the queue busy. Four later sessions (one maximised by the user) did not hit it. Pre-existing code path; reported |

## M5b — viewer extras (walls, crossfade, live cloud, draft mode; branch m5b-viewer-extras)

Checked 2026-10-14 through the hidden `--screenshot` path (the same `Viewer` + `Painter` code as the
window, headless, 1280×800) with the new start-up flags — **not** by driving an interactive window:
no desktop control was available to the implementer, and the user's own window was left alone.
"Headless run" = `app --screenshot … --play-ms n` with the stated flags and its "play summary" line.

| Item | Status | Evidence |
|---|---|---|
| Container walls: drawn as a faint blue inner box; panel checkbox + half-width slider (Bohr, Å, max) | Headless screenshot + unit tests | GPU H₂ `--walls 4`: box at ±4 inside the yellow cell, panel "container walls ☑ 4.00 Bohr — box ±4.00 Bohr = ±2.12 Å (max 5.12 Bohr)", row "of which walls (piston) +0.000000 Ha"; `overlay::container_walls_draw_an_inner_box`, `ui::wall_defaults_fit_the_cell` |
| Walls bounce a moving molecule, E conserved flat | Headless run + `sim` tests | estimated H₂ 49³ translating at 2.2e-3 Bohr/a.u. (`moving_h2.scene`), `--walls 3`, 8 s at speed 1000 (264 ps, 1.06e6 steps): farthest coordinate 2.20 Bohr (the soft layer starts at 2.0), never near the cell max 5.25, no error; drift 1.7e-5 Ha/ps (refit jumps booked); the translation was turned into vibration/rotation by the bounces (T 500–1900 K). `a_free_nucleus_bounces_off_the_wall_with_e_conserved_flat` (306 bounces, \|K/E − 1\| ≤ 1.05e-3), `the_wall_stops_a_one_hartree_nucleus_inside_the_margin` |
| Shrinking the box while playing (piston), its work row | `sim` test; slider not driven | `shrinking_the_box_while_playing_books_positive_work` (W 8.6e-4 Ha = K, E conserved within 4.4e-6). The slider while playing: code reading only; no script yet (`action walls on X` is replayable) — re-check by hand |
| Density crossfade with smooth motion | Render + unit tests; not judged by eye | `crossfade_draws_the_blended_density` (mix 0.5 = ½(a + b) at 0/255, both sampling paths and modes), `cpu_densities_ping_pong_and_snap_on_a_new_grid`, `density_alpha_follows_the_arrivals_and_snaps`. GPU H₂ headless run renders normally (cloud on the molecule). Whether the cloud *looks* smoother in motion: re-check by hand (toggle "smooth motion") |
| Estimated: audit densities and refreshes published, labelled | Worker test + headless runs | `an_audit_and_a_refresh_publish_their_densities_and_the_atoms_their_profiles` (41³: audit density stamped t = 55.25 fs, refresh at 120 fs; H profile 0.997 e). Headless runs: 16 side densities in 8 s (audits only), 18 with `--cloud-refresh 1` |
| Estimated: atomic superposition follows the nuclei, labelled "atomic superposition — not solved" | Headless screenshots + viewer test + unit tests | both estimated stills show the label at the bottom of the view and the cloud on the visible nucleus (the hot-H₂ still: cloud centred on the sphere); `estimated_radio_…` (superposition shown, weight 0.17, blended 83 % with the audit density), `superposition_integrates_to_the_electrons_and_peaks_at_the_nuclei` |
| Draft mode: toggle, red panel line, red banner while extrapolating, last audit "model off by X Ha at step N" | Headless run + still + tests | `hot_h2.scene` (H₂ given 0.26 Ha along the bond: dissociates) `--draft --walls 3`: banner on 98 snapshots during 6 s; panel "guess instead of falling back ☑" with the red "draft mode: forces may be educated guesses" line; "last audit: passed: model off by −3.09e-4 Ha at step 733111 (audit lag 53889 steps)"; the dissociated atoms bounced off the walls (farthest 2.29 Bohr, d(0,1) up to 5.2 Bohr). The final still happened to be on a non-extrapolating step, so the banner itself is not in the picture; `the_draft_banner_counts_the_atoms_beyond_the_envelope`. **Observed**: in this run the drift readout went red (−6e-4 … −1.5e-3 Ha/ps at T 9 000–38 000 K) and "model refit" booked −0.08 Ha — draft mode learning dissociated geometries at extreme temperature; not investigated further |
| Draft mode: atoms tinted by uncertainty | Unit test only | `the_risk_tint_starts_at_half_the_bound_and_is_capped`; in the stills the spheres were not visibly red (the last step was inside the envelope). Re-check by hand: drag an atom far apart with draft on |

## UI redesign (2026-10-14, branch `ui-redesign`)

The side panel became the mockup's chrome (`docs/superpowers/notes/2026-10-14-ui-mockup.html`;
row-by-row record: `2026-10-14-ui-redesign-checklist.md`, 148/148). Evidence levels as above;
"Headless shot" = the hidden `--screenshot` flag (it now renders the whole chrome at 1400×900 and
takes `--size WxH` and `--ui "menu=…;tab=…;select=…;tool=…;cursor=x,y;search=…;thermostat=…;wait=sampling;drag=…;pull=…"`),
"UI test" = `ui::tests` (egui run headless, no window: each control path emits the old command;
every layout and menu draws without input and sends nothing).

| Item | Status | Evidence |
|---|---|---|
| Window opens with the new chrome, no panic | In-window (scripted) | h2o, 10 s, closed with `WM_CLOSE`; log shows the usual start lines only |
| Main layout: top bar, toolbar groups, dock tabs, status chip, atom card, density label, quick switch, gizmo | Headless shot | `docs/images/ui_main.png`, `ui_water.png`, `ui_dock_*.png` |
| Every menu open (File, Scene, View, Physics, Help) | Headless shot + UI test | `ui_menu_*.png`; `every_layout_draws_and_sends_nothing_without_input` |
| Sampling state: card above the toolbar, Play / Step disabled, Model tab progress | Headless shot | `ui_sampling.png` |
| Compact layout (< 1240 pt), dock folded, Place mode guides | Headless shot | `ui_compact.png` (1000×620) |
| Ctrl K search, Enter reveals (dock tab opens), Esc deselects | UI test + headless shot | `ctrl_k_reveals_and_esc_deselects`; `ui_search.png` |
| Each action sends the old Command / DynCommand (the `Action` → command layer) | UI tests | `ui::tests`, one per path (incl. two-click Forget / Start anyway, sidecar load); this tests `actions::apply`, not the widgets |
| Widget → Action wiring, clicked through egui (pointer events, headless) | UI tests (fix round 1) | `clicking_the_toolbar_…`, `clicking_menu_controls_…`, `clicking_the_model_tab_…`: Play, watchable, drags on the speed field (SetSpeed only) and the dt field (SetDt only), an element chip (→ Place), the three solver segments, Delete; the Physics menu's thermostat, gravity g / off and COM; Scene ▸ Apply grid; Forget and Sample with their second clicks. The remaining widgets (other menu fields, the plan fields, the compact flyout, search clicks) are checked by code reading only (review 2026-10-14) |
| The 3D view renders into the central viewport; mouse paths map through it | Unit + render + viewer tests (fix round 1) + headless shots | `render::camera` viewport round trip, `tools` centre unproject / offset pick, `render` offset-viewport render (centre lit, nothing outside), `viewer` chrome rect; `ui_main.png`, `ui_water.png`, `ui_compact.png`: the molecule centred in the view, panel fill (no gap) up to the panels' edges. The mouse paths through the viewport: every replay script (view coordinates are relative to this rect; `replay::tests::view_coordinates_survive_a_window_size_change` picks and places through two different rects). Real OS mouse in a window: not exercised |
| Old worker / viewer tests (backend switch, panel controls, estimated radio, autosave) | Viewer + worker tests | `cargo test -p app --release`: 75 passed, 2 ignored diagnostics |
| Clicking menus, sliders and fields inside an open menu (it must stay open), the compact "☰ Menu" submenus, the element flyout, dock resize, tooltips and strip peeks, toasts timing out, Space with a toolbar button focused, Esc with a menu open | **Not exercised in a window** | need a mouse (posted mouse input does not reach egui); re-check by hand |
| M5b controls re-homed after the merge of `main` (walls + half-width under Scene ▸ Container, the cloud options under View ▸ Density, draft mode under Physics ▸ Estimated solver and in the Model tab, the red banner as a view overlay, the atom tint, the "of which walls (piston)" Readouts row) | UI tests + headless shots | `m5b_actions_send_walls_cloud_and_draft` (the commands `main`'s panel sent), `clicking_the_m5b_controls_sends_their_commands` (walls checkbox, cloud refresh, draft clicked through egui); `ui_menu_scene.png`, `ui_menu_view.png`, `ui_menu_physics.png`, `ui_main.png`. In a window: not exercised |
| Drag pull (user request 2026-10-14): the wheel during a Select-drag sets the drag spring's force cap (×1.25 per notch, 0.01–1 Ha/Bohr) instead of zooming; label by the cursor; TB ▸ Tool sub-line; Physics ▸ Drag pull slider + Reset; Ctrl K "pull" | Viewer + UI + worker + driver tests + headless shot | `the_wheel_sets_the_pull_while_dragging_and_zooms_otherwise` (`Viewer::wheel`: no zoom during a drag, the queued action sent with the next frame, the pull outlives the drag), `the_drag_pull_actions_scale_clamp_and_send`, `a_queued_wheel_action_is_sent_with_the_next_frame`, `clicking_the_drag_pull_reset_and_finding_it`, `the_drag_pull_reaches_the_driver_while_playing` (worker, Morse H₂), the `sim` driver tests (booking, cap, range); `--ui "tool=select;drag=0;cursor=640,420;pull=0.16"` shows the label, `ui_menu_physics.png` the menu. The winit wheel event → `Event::Wheel` conversion: `replay::tests::viewer_input_is_the_only_path`; a drag with the wheel through `Viewer::input`: replay `surrogate_far_drag_recovers` (`wheel 5` mid-drag). A real wheel in a window: not exercised |
| M6 tools in the chrome (Task 10): bond lines (stale fade + label), molecule labels and list, the hover badge, Relax / Stop with its card, the selection's isotope, the periodic table with search and the isotope row | UI + viewer + worker tests + headless shots | `m6_actions_send_relax_isotope_and_toggle_the_view`, `clicking_the_m6_controls_sends_their_commands` (TB Relax/Stop, Physics ▸ Relax/Stop, View ▸ bonds, Selection ²H, table cells O / greyed Ar / ²H), `the_relax_readout_…`, `the_stale_bonds_are_labelled_…`, `pair_lines_…`, `ui::ptable` tests (positions, enabled set, search), `overlay::bond_lines_follow_current_positions_with_the_stale_fade`, `viewer::the_badge_follows_the_picked_nucleus`, worker `labels_are_computed_once_per_snapshot_on_the_worker` (H₂ dihydrogen + one bond drawn; He₂ none), the water test's two drawn bonds and label, `the_isotope_choice_round_trips_through_the_worker`; `m6_water_bonds.png`, `ui_periodic_table.png`, `ui_menu_physics.png`, `ui_menu_view.png` (`m6_water_bonds.png` is also the Selection tab's shot). Replays `labels_and_bonds_h2o` (bonds 2, H2O, ²H on a selected H, Relax converges), `place_three_build_water`, `isotope_swap`, `relax_h2`. **In a window: not exercised** (hover with a real pointer is display only — no script) |

## M6 Task 8 — input replay (2026-10-14)

The viewer's mouse items now run as scripts (`crates/app/replays/*.replay`; format and rules in
`crates/app/src/replay.rs`). Each script is replayed headlessly through the screenshot harness —
the same `Viewer::input` the window's winit events go through, the chrome's `ui::Action`s through
`Viewer::apply` — and asserts on snapshot quantities with the tolerance written in its `expect`
lines; dynamics advance only by `wait steps` (never wall time). `cargo test -p app --release`
runs them all (`replays_pass`, GPU). The scripts were **authored** from the default camera's
geometry (the view points are computed, see each script's header), not recorded by hand: no
interactive window was available to the implementer. `--record` itself is covered by
`replay::tests::a_recorded_session_replays_to_the_same_snapshot_sequence` (a live session through
`Viewer::input`/`apply` on an analytic worker, recorded, replayed: the same snapshots at every
`wait idle`). Wall times: `replays_pass` in the run of record (RX 7900 XTX, another GPU process
running), each including its viewer's start and first solve; 262 s in all, most of it the O
free-atom reference (≈ 50 s at 49³, ≈ 70 s at 57³), the water relax (≈ 60 s at 57³: 0.16 Ha to the
near-linear minimum of the σ = 0.25 surface) and the surrogate's sampling pass (≈ 35 s).

| Script | Items it covers | Wall |
|---|---|---|
| `camera_leaves_scene_alone` | M2 orbit / pan / zoom (the input path; nothing sent, nothing selected) | 3.4 s |
| `place_three_build_water` | M2 Place on the focus plane; M6 build water: 2 bonds, "H₂O" | 58.0 s |
| `select_drag_paused_teleport` | M2 Select + drag; M4 paused drag = teleport; fix round 1 I1 | 2.7 s |
| `select_drag_playing_spring` | M4 drag while playing = spring, no jump (R12 cap, I1) | 3.0 s |
| `delete_and_space` | M2 Delete key; M4 Play / Pause with Space; review m2 (a select-click sends nothing) | 2.6 s |
| `pin_holds` | M4 Pin (two pinned nuclei keep their distance while the third moves); Place | 4.3 s |
| `isotope_swap` | M6 isotope of the selection (²H), the run plays on it | 4.1 s |
| `relax_h2` | M6 Relax converges to the σ = 0.25 minimum (1.75–1.83 Bohr) | 3.6 s |
| `labels_and_bonds_h2o` | M6 bonds and label on water, ²H on a selected H, Relax converges, still water | 128.7 s |
| `surrogate_far_drag_recovers` | M5 far drag → fallbacks → the model answers again (nucleus 1 pinned so the pull stretches the bond: R ≈ 3.1 Bohr, two fallbacks kept as samples, epoch 1 → 3, then Surrogate at R up to 3.39); the wheel sets the pull mid-drag | 51.8 s |

**Found by the scripts:** a relax in the window stopped after its first FIRE step — the worker did
not count a running relax as work, so it waited for the next command (any click resumed it for
one more step). `worker::has_work` now includes it; `a_relax_runs_to_its_end_without_another_command`
(worker test, a 20 ms analytic provider) pins it, `relax_h2` and `labels_and_bonds_h2o` check it on
the GPU.

**Still manual** (no script, and why): anything judged by eye — smooth motion and the density
crossfade, the distance guides, the hover badge, the draft-mode tint, the spheres' and cloud's
look; panel layout and widget behaviour — menus staying open, sliders and fields inside menus,
the compact flyouts, dock resize, tooltips, toasts timing out, Space with a button focused, Esc
with a menu open, the readouts' text; actions whose meaning is panel state a script does not
carry — Apply grid, scene and model files, Sample / "Start anyway" with the plan fields; window
mechanics — resize / minimise / close, the OS mouse and wheel themselves (the winit → `Event`
conversion is unit-tested); error expectations (a script has no `expect error` yet: coincident
placement, the cell-face pause).
