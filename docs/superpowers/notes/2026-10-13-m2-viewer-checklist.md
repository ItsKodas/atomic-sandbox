# M2 viewer — manual checklist status

The viewer's manual checklist is Task 7, Step 5 of `docs/superpowers/plans/2026-10-13-m2-viewer.md`.
This note records how far each item was verified when M2 landed. Until M5 adds the
input-replay harness, these items have no automated coverage beyond the unit tests named here.
Re-run the unverified ones by hand after UI or render changes: `cargo run -p app --release`.

"In-window" means it was seen in a running window: a capture or the stderr log, from the
implementer's own instance or the user's live session. "Code reading" means the logic was
checked (and unit-tested where noted), but nobody exercised it in a window.

| Item | Status | Evidence |
|---|---|---|
| Window opens (RX 7900 XTX, Vulkan, `Rgba8UnormSrgb` surface) | In-window | log + window capture |
| H₂ appears with its density; snapshot uploads with right dims | In-window | 49³ × 0.25 Bohr, ρmax 0.2231, E −0.962716 Ha, 8 SCF steps, ~5.9 s; capture shows density + spheres |
| Resize | In-window (user) | maximised to 2560×1369; correct rendering afterwards, no panic |
| Resize / Outdated while the worker submits on the shared device | Headless test + code reading | Seen once (2026-10-14): wgpu 30 panicked ~2 s after launch with "Failed to wait for GPU to come idle before reconfiguring the Surface" (`ConfigureSurfaceError::GpuWaitTimeout`, a validation error wgpu-core raises when another thread submitted during `configure`'s idle wait). `SurfaceState::resize` now captures it in a thread-local error scope and returns `Err` leaving the old surface; `Gfx::resize` logs "surface: reconfigure … deferred to the next frame" and `redraw` retries while `config` differs from the window size. `crates/app/tests/surface_reconfigure.rs` recreates the race (a worker thread submitting with readback waits, 100 reconfigures of a hidden window, ~3 s): without the scope it panics at the reported wgpu line; with it 13–25 % are refused and the retry succeeds. Manual check: launch, press Play, resize/maximise repeatedly for ~10 s: at most the deferred line on stderr, never a panic |
| Minimise | Code reading | `SurfaceState::resize` ignores 0 sizes; `redraw` returns early at 0×0; `about_to_wait` stops requesting redraws while minimised (polls instead) |
| Preset switching (He, H₂O) | In-window (user) | He → E −2.138685; H₂O → spheres update at once, spinner shown, old density kept meanwhile |
| Isosurface mode | In-window (user) | capture |
| Solver error → red message, old picture stays | In-window (user) | charge > protons on H₂ → red "exceeds the 2 protons" line, previous density still drawn |
| Gamma slider → volume updates | In-window (user) | capture at gamma 0.55 |
| Orbit / pan / zoom | Code reading | `render::Camera::orbit/pan/zoom`; `Viewer::zoom` floors the distance at 2.0 |
| Place adds a nucleus on the focus plane, density updates | Code reading | `tools::place_point` unit-tested |
| Display aids: cell box, floor grid (xz, +y up) with axes, placement-plane grid in Place mode, 1 Bohr / 1 Å spacing | Code reading + headless | `render::lines` pass tested headless (lit pixels, occluded by spheres, no depth write); `overlay::build` unit-tested |
| Distance guides: faint lines + Bohr/Å labels from a dragged nucleus (or the Place cursor) to nuclei within 6 Bohr | Code reading | `tools::distance_guides` / `tools::project` unit-tested; overlay in `Viewer::distance_overlay` (egui foreground layer) |
| Select + drag, Delete key | Code reading | `tools::pick` / `tools::drag_point` unit-tested; drag sends `MoveNucleus` per motion event, the worker coalesces (`sim::queued_commands_are_coalesced_into_one_solve`) |
| Coincident placement → red "coincide" error | Code reading | worker path: `sim::provider_failure_pauses_with_message` (M4; was `coincident_nuclei_report_an_error_not_a_crash`); UI path is the same as the charge errors above |
| Apply grid 33/0.3 recreates the volume | Code reading | upload recreates on `volume_dims != [n;3]` (also hit by H₂ 49³ → H₂O 57³); worker: `sim::grid_change_resizes_the_density` |
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
| Play / Pause with Space | In-window | the play log starts at once; Pause logs an idle snapshot within one step. Once, a posted Space did not take and the next one did |
| Vibration | In-window | d(0,1) runs 1.600 ↔ 2.017 Bohr. Minima at t = 0, 20.75 and 52.0 fs give a period of **10.4 fs**. T runs 0 ↔ 890 K. E_total stays in −0.980640 … −0.980669 Ha, a Verlet swing of 3e-5 Ha (≈ (ω dt)²/4 of the 4e-3 Ha exchanged). After 100 steps the drift is +8e-5 … −7e-7 Ha/ps, ~1e-5 typical |
| Speed of the run | In-window | 2.2–3.5 fs/s at 77–120 ms per step (4–7 SCF steps, ~240–280 readbacks), 49³ |
| Frame time while playing | In-window | interval p50 **4.2 ms**, p95 5.1–5.8 ms, max 6–38 ms per 5 s window. The 30–38 ms maxima fall in the first window after Play |
| Readouts and sparkline | In-window | captures show T, KE, E_KS, E total, E conserved (Ha and eV), drift, SCF steps / readbacks / ms per step, fs/s and steps per tick, and the E_total/T sparkline in anti-phase. The drift turned red in the runaway below. Amber was seen in the log only (+1.07e-4 Ha/ps), not in a capture |
| Play / Select / orbit with the mouse | In-window (user) | the user's session: Play, the Select tool, a camera orbit and a zoom |
| Drag while playing = spring | In-window (user) + test | user log: E_conserved separates from E_total by the spring energy while the drag is held, and T spikes to 2.4e4 K. `drag_while_playing_uses_the_spring_not_teleport`: max \|Δr\| per snapshot 0.021 Bohr, no jump |
| Select + drag while paused = teleport | In-window (user) | user log: an idle snapshot after the move, d(0,1) 4.19 → 2.45 Bohr, 11 SCF steps |
| Provider failure pauses with a message | In-window + test | GPU: "SCF did not converge after 200 iterations (Δρ = 1.9e-4)" at d = 2.61 Bohr during the user's run paused it with the red line; Space resumed. `provider_failure_pauses_with_message`: "nuclei 0 and 1 coincide" on both backends |
| CPU backend playing | In-window (`--backend cpu`) | 2.4–3.6 s per step, 0.06–0.11 fs/s. The trajectory matches the GPU run at step 21 (d(0,1) 2.0167 Bohr, T 0.2 K) and step 25 (1.9838 Bohr, 220.8 K). Frame interval p95 4.4–4.9 ms during CPU steps |
| Close during a CPU MD step | In-window (scripted `WM_CLOSE`) | the process exited in 142 ms |
| CPU ↔ GPU radio | Headless | `switching_backend_respawns_the_worker_with_the_current_scene` (the panel's settings are re-sent; the run restarts paused) |
| Pin, Rewind, thermostat, gravity preset | Headless | `panel_controls_reach_the_driver`: the pinned nucleus is bit-still over 5 steps; Rewind flips every velocity exactly; switching the thermostat on at rest gives T = 300 K and the Maxwell–Boltzmann note; g = 1.0836e-22 Bohr/a.u.² reaches the driver |
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
does not reach egui; re-check them by hand.

| Item | Status | Evidence |
|---|---|---|
| A nucleus closer than 3h to a face, or outside the cell, pauses with "left the simulation cell" (R12) | Unit + worker test | provider `nuclei_near_or_outside_the_cell_face_are_refused` (CPU and GPU, refused before any solve); app `a_nucleus_at_the_cell_face_pauses_with_a_message` |
| Drag spring force capped at 0.1 Ha/Bohr, work booked exactly (R12) | Unit test | `drag_force_is_capped_and_its_work_booked`: the first step is exactly F_max; E_total peaks at the capped potential, 0.18994 vs 0.19000 Ha (uncapped would be 1 Ha); \|ΔE_conserved\| ≤ 1.3e-3 Ha |
| "work done by tools" readout (E_total − E_conserved) | Code | `ui::dynamics_readouts`, with a tooltip on the drag-speed trade-off |
| A paused teleport warm-starts (R13) | Worker test | `moving_a_nucleus_warm_starts` (h2 49³, 0.02 Bohr): warm 381 readbacks / 7 steps vs cold 477 / 8, ratio 0.80 < 0.85; ΔE 1e-8 Ha |
| The drag mode follows Play/Pause at every motion event, never a teleport while playing (I1) | Unit test | `tools::drag_mode_follows_the_play_state_at_each_motion`. A select-click without motion no longer tethers the nucleus (review m2) |
| Drag targets, teleports and placements clamped into the cell minus 3h and ½h (I2) | Unit test | `tools::clamp_keeps_positions_inside_the_validation_margin`. The marker draws the clamped target |
| Focus loss or the cursor leaving the window while a button is held counts as a release (m1) | Code | `App::release_all` (`WindowEvent::Focused(false)`, `CursorLeft`) |
| Atom card (M4+, user request): selecting a nucleus shows a top-right Bohr-style schematic with Z protons, N neutrons (most abundant isotope), electron shells, and the not-to-scale / shared-electrons caveat | Code reading + unit tests (`atom_card::tests`) | `crates/app/src/atom_card.rs`, hooked in `Viewer::run_ui` |

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
| Forget model (two clicks within 4 s), Sample more, audit field, sampling-plan fields, pin / thermostat under the surrogate | Code reading only | need a mouse; re-check by hand. Pins and the thermostat are driver features independent of the provider (M4 tests). The first pass starts at once when estimated is picked — the panel's plan if it fits about 2 minutes, else scaled down, else an offer with "Start anyway" (Task 9); edit the plan by Cancel → change → Start |
| Fix round 1: smooth motion (nuclei interpolated between the last two playing snapshots) | Unit test + in-window (frame rate only) | `interpolation_is_monotone_and_snaps_on_count_change`. In the window the frames stayed at p50 4.2 ms / p95 5.9–6.4 ms with it on; whether the motion *looks* smooth was not judged by eye from the log or the stills — re-check by hand (toggle "smooth motion" in the display row) |
| Fix round 1: tick pacing and "watchable" | Worker test + unit test + in-window | speed 1 on the surrogate: 15.7 fs/s (0.25 fs per 16 ms; 8.3 before the fixed-schedule deadlines, Windows timer granularity); `the_watchable_speed_…`: 1 step/tick, 15.6 fs/s at dt 0.25 on the surrogate, 2.5 fs/s on DFT. In the window the button read "watchable (≈ 1 fs/s)" during the user's far drag (every step a 250 ms fallback) |
| Fix round 1: a far drag in the window (user) | In-window (user) | the user dragged H₂ apart while playing on the surrogate: fallbacks 45 % of 100, T 3600 K, 250 ms/step; the refit then **untrusted** the model ("max training energy residual 8.6e-3 > 1.0e-3 Ha/atom", 322 samples) — Task 7's "one stray sample can untrust a model", now seen in a window; the panel showed the reason |
| Fix rounds 3–4: once it falls back it comes back (the user's report: 100 % fallbacks, "3540 samples epoch 0 untrusted") | Worker + `sim` tests | root cause: one sample the model cannot fit failed the training-residual trust rule at every later refit, and every fallback kept its sample. The sample in the user's session was H₂ pulled to R = 8 Bohr in the 10 Bohr box (41³) — +0.3 Ha of **box confinement** (its nuclei 1 Bohr from the faces; fix round 4 corrected the round-3 "beyond the cutoff / restricted singlet" story). Now: a degree-cap retry, then quarantine (ruling S15), and box-affected answers are not trained on. App tests: `after_a_far_drag_the_surrogate_answers_again` (R = 8 → not kept, R = 3.3 → Surrogate again within the asserted ≤ 20 fallback steps and ≤ 2 refits; the test prints the count — 1 in the Task 8 round-4 report, 15 with 0 refits since the final fix wave (deterministic over three runs): the 41³ pass's stretched scan end is box-affected, now flagged as such and left out of the fit (final review I1), so R = 3.3 starts outside the trained envelope and the box-affected answers there are not kept — the model answers once the vibration brings the bond back in), `a_far_geometry_in_a_small_box_falls_back_and_is_not_trained_on`; `sim` tests with analytic teachers for the quarantine itself: `a_stray_sample_is_quarantined_then_superseded_and_trust_returns`, `a_quarantined_sample_is_released_when_the_model_reproduces_it`, `a_molecule_parked_beyond_the_cutoff_neither_grows_the_store_nor_untrusts_the_model`, `a_box_affected_answer_is_used_but_not_trained_on`. Not re-checked in a window — re-check by hand: drag H₂ far apart while playing, then let go |
| Task 9: "trusted, Fallback, 100 % of 100, teacher gap +2.9e-3 Ha" near equilibrium (the user's round-3 log) | Worker tests | root cause: box confinement (≈ 50 Ha per e/Bohr³ of face density) under the old 1e-3 box rule — an audit near a face failed on +6.4e-3 Ha of confinement, the refit quarantined that sample, and its descriptor-space fence covered the equilibrium bond everywhere in the cell. `a_molecule_parked_near_a_face_is_not_fenced_by_the_box` (H₂ 41³ parked 1.5 Bohr off-centre, audits every 100: 0/251 fallbacks, the audits "not judged (box-affected)", 0 quarantined; back at the centre Surrogate after 1 step, next audit −5.3e-5 Ha); `after_a_hot_excursion_the_model_answers_at_equilibrium_again` (12 random hot teleports, then equilibrium: Surrogate after 1 step, next audit −5.4e-5 Ha; with the old 1e-3 rule the model ended untrusted, 9 quarantined). In a window: not yet |
| Task 9: audits in the background (ruling S14) | Worker tests + headless screenshot | the audit step is the model's; a second GpuKs solves it meanwhile ("last audit: in flight on the background teacher", then "passed: −5.3e-5 Ha (274 ms DFT, judged 17 steps after dispatch)"). H₂ 41³ at audit 2000: **38 816 fs/s** (Task 8: 3 474), audits off 43 561; headless 49³ screenshot run 39 841–41 969 fs/s over 2 s. `sim`: `a_background_audit_runs_while_the_model_answers`, `a_failed_background_audit_untrusts_and_books_the_flip` (ΔE_c across the flip ≤ 1e-5). In-window fs/s: not re-measured |
| Task 9: sampling-pass ETA, 2-minute first pass, "Start anyway" | Worker + unit tests | `the_first_pass_keeps_to_its_time_target_or_asks` (H₂ default plan as is at 0.12 s a solve, scaled to 133 solves at 0.9 s, an offer at H₂O 57³ 2.4 s: 210 solves ≈ 504 s); `pause_and_cancel_…` (H₂O 97³: the automatic pass is an offer — 210 solves ≈ 800 s — and "Start anyway" starts it); the button's "N solves, ≈ T" and the progress bar's time left seen in the headless screenshot ("273 solves, ≈ 24 s"). The two-click "Start anyway" on the manual button: code reading only |
| Task 9: readouts — model / DFT step costs, "model refit" and "model force mismatch" rows, "yes, but not answering here" | Headless screenshot + `sim` tests | `a_slow_dft_step_does_not_throttle_the_model_steps_after_it` (400 steps in the tick after a 100 ms fallback; [399, 1, …] with one mixed average), `a_trusted_model_that_keeps_falling_back_says_so`; the rows in `docs/images/surrogate_h2.png` |
| Fix round 1: leaving estimated never blocks the window; the session autosave survives quick toggles and a refused autoload | Worker + viewer tests | `leaving_the_estimated_solver_neither_blocks_nor_loses_the_autosave` (switches during a cold 97³ evaluation: worst `run_ui` 5.2 ms, file unchanged), `a_worker_shut_down_before_ensure_model_leaves_the_autosave_alone`, the refused-autoload retire in `model_sidecar_saves_and_loads_with_the_scene` |
| Crash seen once at start-up (M5) | In-window, not reproduced | the first `--estimated` session panicked 2 s in, while sampling, in `Surface::configure` ("Failed to wait for GPU to come idle before reconfiguring the Surface") — the resize path of `gpu::SurfaceState::resize` while the worker kept the queue busy. Four later sessions (one maximised by the user) did not hit it. Pre-existing code path; reported |
