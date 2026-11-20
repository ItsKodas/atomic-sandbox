# Trust tightening: the BEST-collection go/no-go (2026-11-19, plan Task 0)

The read-only report the spec (2026-11-18 §6) asks for before any trust change is built. It previews the
evidence floor's effect **at load** on the user's BEST collection.

## How it was measured

- **The collection:** `models/collections/lsda_h0.15_s0.15_tol1e-6_kohn-sham-gpu_all-54eedc40ddc5`
  (1657 samples, 434 history points).
- **Read-only.** It was copied without its `.lock` into a scratchpad folder, and only the copy was opened,
  attached and refitted.
  - The original's 9 files keep their SHA-256, size and mtime, checked before and after.
  - The test itself also asserts that sizes, mtimes and content hashes are unchanged (the `.lock` is never
    opened).
- **CPU only, no teacher.** There is no GPU context. The teacher refuses every call, and it was called 0
  times, so nothing was solved. The attach and refit took 62.6 s.
  - The refit gave σ_F 1.29e-3 and Q′ 1902, under `FallbackPolicy::default()`.
- **Command:** `cargo test -p validate --release --test surrogate trust_tightening_go_no_go_report -- --ignored --nocapture`
  (`BEST_COPY_DIR` sets where the copy goes).
- **"Today"** is the verdict of a fresh attach: no standing verdict, so the entry limits apply. That is
  what a load does today.
- **"At load"** is the new rule.
  - Saved points carry no fit id, so n_fit = 0 and the floor (12 fits, 60 points) is never met.
  - A key trusted today on evidence ≥ 3 therefore also needs LOO force rms, force max and energy max all
    ≤ 1× (`class_refusal`).
  - A key with evidence < 3 keeps today's rule.
- **n_pts (gens)** counts the key's attributed audits and class fallbacks (codes 5/6) over the whole
  history, with their distinct `model_gen`s. It is information only; the verdict reads `evidence`, which
  is windowed.
- **Scenes** are the collection's own scene names whose samples hold the key, with the sample counts.

## The table

| class | key | trusted today | refusal today | evidence | n_fit | n_pts (gens) | LOO F rms | LOO F max | LOO E max | λ_sum | at load | why | scenes (samples) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pair | around H, with H | yes | – | 38 | 0 | 38 (33) | 0.253 | 1.017 | 0.947 | 37.30 | **no** | floor not met (n_fit 0); LOO force max 1.02× > 1×: LOSES trust | H2 (923), H2 (held out) (21), H3 (4) |
| many-body | around H, with H | no | LooFloor 1.057 / 0.707 | 3 | 0 | 40 (15) | 1.057 | 1.322 | 0.586 | 1.45 | **no** | untrusted today (unchanged) | H3 (161), H3 (held out) (10) |
| many-body | around H, with H and O | yes | – | 3 | 0 | 3 (3) | 0.150 | 2.506 | 0.511 | 1.41 | **no** | floor not met (n_fit 0); LOO force max 2.51× > 1×: LOSES trust | H2O (319), H2O (held out) (2) |
| many-body | around H, with H and P | no | Samples 217 / 238 | 2 | 0 | 2 (2) | 0.099 | 0.756 | 0.300 | 1.00 | **no** | untrusted today (unchanged) | H3P (217) |
| many-body | around O, with H | yes | – | 3 | 0 | 3 (3) | 0.094 | 0.764 | 0.511 | 1.41 | yes | floor not met (n_fit 0); LOO ≤ 1×: keeps trust | H2O (319), H2O (held out) (2) |
| many-body | around P, with H | yes | – | 2 | 0 | 2 (2) | 0.043 | 0.208 | 0.300 | 1.00 | yes | evidence 2 < 3: today's rule | H3P (217) |

**Keys 6. Trusted today 4. Trusted under the new rule at load 2. Two keys lose trust.**

## The keys that lose trust

1. **pair, around H, with H.**
   - Today's evidence is 38 (38 points over 33 generations, none with a fit id).
   - LOO force max 1.017×, rms 0.25×, energy max 0.95×.
   - Scenes: H₂ (923 samples), and the 4 H₃ samples where an end atom has one neighbour.
   - **This is the knife edge §3 predicted.** The key has the most evidence in the collection and is
     calibrated today. It loses trust only because its single worst LOO atom is 1.7 % over the bound.
2. **many-body, around H, with H and O.**
   - Today's evidence is 3 (3 points, 3 generations).
   - LOO force max 2.51×, rms 0.15×, energy max 0.51×.
   - Scene: H₂O (319 samples).
   - It refuses the whole water molecule, even though its O key ("around O, with H") keeps trust.

## The user's typical scenes

- **H₂: affected.** It needs the pair key, which loses trust.
- **H₂O: affected.** Its H atoms' key loses trust. The O key alone would still pass.
- **H₃ (neutral, in this collection): unchanged at the many-body level.** Its H key is already untrusted
  today (LooFloor 1.057). A stretched H₃ whose end atom sees one neighbour also reads the pair key,
  which now refuses too.
- **H₃⁺ and NH₄⁺: not in this collection.** Charged scenes keep per-scene models, and the collection holds
  no N key.
- **PH₃ (H3P): unchanged.** The P key keeps today's rule at evidence 2, and the H key is already refused
  on samples (217 of 238).

## Caveats

- **Only the floor at load is previewed.** λ_T and λ_E cannot be previewed on legacy points: they carry
  no flags, so both start at 1. Rules 6 and 7 can only refuse more once fresh points arrive.
- **The way back is the existing one** (spec §3, "the dead end"): the teach prompt after
  `TEACH_PROMPT_STREAK` class fallbacks, or moving the scene.
  - For the pair key a single blocker sample decides, since the max is 1.017×.
  - A parked H₂ gains no fits while its fallbacks are thinned.
- **Whether to proceed is the user's call** (spec §6; plan Task 0 step 6).
