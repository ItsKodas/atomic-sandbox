# Windows installer and in-app updater (2026-11-12)

User request, 2026-11-12: "Automatic updates within app, as well as being able to choose previous versions, betas etc. All using GitHub's release systems", plus "an actual installer". Linux was dropped for now because the user can't test it ("maybe we drop linux support for now, since i cant easily test it").

## Intent

Atomic Sandbox is public (`github.com/ItsKodas/atomic-sandbox`). Today a release is made by hand:
- `build.bat` lays out `dist\AtomicSandbox-<ver>-windows-x64\` and zips it;
- the zip is uploaded to a GitHub release (one so far: `v0.1.0`).

A newcomer has to unzip it and run an exe from wherever they put it. Updating means repeating that by hand.

The user wants:
- a newcomer gets a real install;
- the app keeps itself current;
- anyone can step between versions and betas from inside the app.

**Success, in the user's words:** "download the installer, click next, then it opens on H2". Beyond that, a version change is one click inside the app.

## Goals

1. **A Windows installer on every GitHub release.**
   - Per-user, no admin rights, nothing to choose but "Next".
   - It adds a Start-menu shortcut and an uninstaller in Windows' Apps list.
   - Its last page has a ticked "Launch Atomic Sandbox" box. The app then opens on H₂, paused, with today's defaults.
2. **A portable zip on every release as well**, as today.
3. **User data in the per-user folder.**
   - Profiles, saved models, collections, scenes the user saves, timings, captures and the pseudopotential cache go to `%APPDATA%\Atomic Sandbox`, not beside the exe.
   - On the first launch of an installed build, the user's existing data folders are **copied** in once. The originals are never moved or deleted.
4. **An in-app updater**:
   - over the GitHub Releases API;
   - with a Stable channel and a Beta channel;
   - and able to install any listed version.
5. **A release workflow** (GitHub Actions) that builds the installer and the zip when a version tag is pushed.

## Non-goals

- Linux, macOS, and any non-Windows installer. The existing `cfg(windows)` fences stay so that Linux can be picked up later.
- Code signing with a paid certificate. Releases are unsigned at first; the SmartScreen warning is documented in the release notes.
- System-wide (all users) installs.
- Telemetry of any kind. The only network traffic is reading GitHub's public releases API and downloading release assets.
- Delta or patch updates. A whole release asset is downloaded each time.

## Anti-goals: it fails even if it "works" when it…

- **…interrupts a solve, a refit or a sampling pass**, or restarts the app without the user's click.
- **…deletes, moves, converts or rewrites user data** while migrating, downgrading or uninstalling. The uninstaller keeps `%APPDATA%\Atomic Sandbox` unless the user ticks "also remove my data".
- **…installs a download that doesn't match its checksum and signature.**
- **…lets an older version silently fail on newer data.** Downgrades warn first.
- **…breaks the developer workflow.** `cargo run -p app --release`, `start.bat`, the tests and the replay scripts must keep using the repo's own `models/`, `profiles/` and `scenes/` exactly as now.

## What the user sees

### Install

1. The user downloads `AtomicSandbox-Setup-<ver>.exe` from the release page.
2. SmartScreen shows "Windows protected your PC" (unsigned), and they click *More info → Run anyway*. The release notes say so in one line.
3. They click Next, Next, then Finish with "Launch Atomic Sandbox" ticked. The app opens on H₂, paused. There is no tour and no welcome screen.

### Updates during a session (no popups, ever)

**When it checks:** quietly at launch, then every 6 hours while the app is open. The interval is my call.

**The chip.** When a newer version exists on the chosen channel, a small chip appears at the right end of the toolbar: **"v0.2.0 available"**. It doesn't interrupt anything.

**The Version panel.** Clicking the chip opens it; it is also under **Help → Version…**. It contains:
- **Channel:** *Stable* (releases only) or *Beta* (pre-releases too).
- **Versions:**
  - every release, newest first, with its date and its release notes (collapsed);
  - the installed one is marked "installed".
- **Install and restart**, on any entry.
  - The download runs in the background, and the panel shows progress.
  - When it's ready, the button becomes **Restart now**.
  - If a refit, a sampling pass or a solve is running, Restart now waits for it, or asks before cancelling it.
  - After the restart, the scene, camera and dock layout come back, the way a saved profile restores them.
- **Check for updates now**, and **Check automatically** (on by default).

**Verification.** Every download is verified against the release's SHA-256 file and its signature before anything is replaced. A failed check means nothing is installed, and the panel shows the reason.

### Downgrades

**What the panel checks.** Choosing an older version first compares that release's data-format manifest (below) with the data in `%APPDATA%\Atomic Sandbox`.

**What it warns.** When the older version can't open some of it, the panel says so and names the folders. For example: "v0.1.0 can't open 2 of your collections (they were written in a newer format). They stay untouched; that version just won't load them."

**What it never does:** delete or convert anything.

## Decided "how"

- **Versions** (the user handed this to me: "Ill let you decide on version tags"):
  - Tags follow semver: `vX.Y.Z` for a release, `vX.Y.Z-beta.N` for a pre-release.
  - The workflow marks a `-beta.N` tag as a GitHub pre-release automatically.
  - The tag must equal the workspace `version` in the root `Cargo.toml`, or the workflow fails.
  - The app's own version comes from `CARGO_PKG_VERSION`.
- **Releases are built by GitHub Actions on a pushed tag** (`windows-latest`, free for a public repo). This is my call: builds are reproducible and don't depend on the user's PC.
  - The workflow builds the app.
  - It lays out the same folder `build.bat` does.
  - It produces the zip and the installer.
  - It writes `SHA256SUMS.txt`, signs it, and writes `manifest.json`.
  - It creates the GitHub release with the assets, as a draft.
- **Physics gates stay on the user's PC.** The Actions runner has no GPU, so the GPU suites can't run there. The release checklist (in `docs/dev-reference.md`) says: run the gates locally on the tagged commit before publishing the draft. The runner may run the CPU-only suites; that's left to the builder.
- **Publishing is always confirmed by the user.** A tag push and a draft becoming a published release are both outward-facing. Claude asks before each.
- **The installer is Inno Setup**, installing per-user to `%LOCALAPPDATA%\Programs\Atomic Sandbox`.
  - Because the install is per-user, the updater can replace files without admin rights.
  - The updater runs the new installer silently (`/VERYSILENT /SUPPRESSMSGBOXES`, with a relaunch flag) once the app has exited. This is my call.
- **The portable zip.**
  - It contains a `portable.txt` marker beside the exe. With the marker, all data stays beside the exe, as today.
  - The updater in a portable copy shows the versions but offers "Download zip" instead of installing. Replacing a portable folder in place is a non-goal.
- **Where data lives** (one function, replacing the beside-exe / current-dir lookups in `crates/app/src/scene.rs` and `capture/mod.rs`):
  1. A `models/` folder beside the exe or in the current directory, which is the dev checkout, or a `portable.txt` beside the exe. Use those locations, exactly as now.
  2. Otherwise, the installed build: `%APPDATA%\Atomic Sandbox\{models, profiles, scenes, timings, captures, …}`.

  The pseudopotential cache follows the same rule. Shipped read-only content (the preset scene library, the bundled trained models) stays in the install folder and is read from there. User saves go to `%APPDATA%`.
- **The one-time migration.**
  - **When it runs:** on the first launch of an installed build, when `%APPDATA%\Atomic Sandbox` doesn't exist yet and an older data folder is found. "Found" means the last portable or dev location, recorded as below, or one the user picks from a prompt that says what it will copy.
  - **What it does:** copies it in. It never moves anything.
  - **What it records:** `migrated_from.txt`.
  - **If a copy fails:** that folder is skipped and named; nothing is half-written.
  - The detail of finding the old folder is left to the builder. A simple "Import data from a folder…" prompt on first launch is acceptable.
- **Signing.**
  - **What gets signed:** `SHA256SUMS.txt`, signed with minisign (Ed25519).
  - **The public key** is compiled into the app.
  - **The private key** is a GitHub Actions secret. **The user creates and adds the secret**: Claude doesn't handle secrets. Claude gives exact commands.
- **The manifest.** `manifest.json` per release holds:
  - the version and the channel;
  - per asset: name, size, sha256;
  - the data-format versions this build reads and writes: collection record versions, sidecar and model-file versions, profile and scene and replay format versions, pseudo cache key.

  Each value is read from the code's existing version constants, not typed by hand. A release without `manifest.json`, such as `v0.1.0`, is treated as "format unknown": downgrades to it warn generically.
- **The updater's network.**
  - HTTPS GET of `api.github.com/repos/ItsKodas/atomic-sandbox/releases`, unauthenticated (60 requests per hour is plenty), and of the release asset URLs. No other host, except GitHub's own asset CDN.
  - **Amended 2026-11-13** (Task 3 review I1; the user's intent was "GitHub only, no telemetry"):
    - GitHub answers every asset URL with a redirect to its download CDN, so the CDN counts as part of "the release asset URLs".
    - Redirects are followed by hand: at most 2 hops, HTTPS only.
    - The allowed hosts are exactly `api.github.com`, `github.com`, `release-assets.githubusercontent.com` and `objects.githubusercontent.com`.
    - Any other host is refused.
  - A plain HTTP client crate (`ureq` or similar, plus `sha2`, `semver`, `minisign-verify`) is plumbing, allowed by CLAUDE.md rule 2.
  - Every network call runs off the UI thread. Offline means a quiet "couldn't check (offline)" in the panel, never an error dialog.
- **Controls go through `ui::Action`.** Check now, install a version and restart are `ui::Action`s, so `--record` sees them. The channel and auto-check toggles are display settings stored in the profile.
- **The README.** Its install section is rewritten (installer first, portable zip second) in the same change, per CLAUDE.md.

## Builder-check rulings (my calls, 2026-11-12)

These supersede the text above where the two differ.

- **The per-user root is the existing `%LOCALAPPDATA%\atomic-sandbox`, not `%APPDATA%\Atomic Sandbox`.**
  - Every build already keeps these there:
    - `current.profile`
    - `timings.txt`
    - `training\` (the pass checkpoints)
    - `cache\pseudo` (with the `ATOMIC_SANDBOX_CACHE_DIR` override)
  - Roaming was incidental; the data is large and machine-local.
  - An installed build adds `models\`, `profiles\`, `scenes\` and `captures\` under the same root. Collections go in `models\collections`.
  - Dev and portable builds keep their beside-exe and current-directory folders for those four, exactly as now. They keep sharing the four existing LOCALAPPDATA items with the installed build.
    - Sharing the pseudo cache saves regeneration.
    - A newcomer has no such folder, so they still open on H₂ with the defaults.
  - Every "`%APPDATA%\Atomic Sandbox`" above now reads `%LOCALAPPDATA%\atomic-sandbox`.
- **Uninstall "also remove my data"** is unticked by default. When ticked, it deletes only what the installed build created: `models\`, `profiles\`, `scenes\` and `captures\` under the root. It never touches `cache\`, `training\`, `current.profile` or `timings.txt`, which dev builds share. The checkbox text says so.
- **Scenes and profiles get no version marker.** The scene parser rejects unknown lines, so a marker would itself break every older build.
  - The manifest records a code-only format number for each, bumped when its format changes.
  - A downgrade across a bump warns by type ("scenes saved since v0.3 may not open in v0.2"), not by file.
  - Collections, model files, history, replays, checkpoints and the pseudo cache are checked from their real version constants, and named by folder.
- **What a restart after an update brings back:**
  - the scene as it stands: positions, velocities, MD time, charge and spin;
  - its settings: backend, quality, ionic, walls;
  - paused (always paused after a restart);
  - the dock layout and display settings (the profile);
  - an attached collection: it is already on disk sample by sample, and re-attaches and refits.

  The scene and settings are written as a session scene file under the root before the installer runs, and loaded on relaunch.
  - **An unsaved in-session model without a collection** is not snapshotted. Restart now says so first and offers Save model…, Restart anyway or Cancel.
- **Release contents are committed files only**, built from a clean checkout on Actions.
  - Personal profiles never ship.
  - `build.bat` switches to committed files for `models\` and `profiles\`, as it already does for `scenes\`, so a local package equals the CI one.
  - Trained models: only the ones already committed ship. The user's untracked test models (`bh3.model`, `nh4+.model`, `ph3.model`) stay local and are never committed (user, 2026-11-12: "dont commit my test training models").

## Left to the builder

- The crate choices (HTTP, zip, signature verification) and the module layout. Suggested: `crates/app/src/update/`.
- Exactly how the app hands off to the installer and relaunches, and how the session state is restored.
- Whether the Actions workflow also runs the CPU-only test suites.
- Tests: the version and channel selection logic, manifest parsing, checksum and signature failure, the downgrade-warning logic, the data-location rule (dev, portable and installed cases), and migration copying (including a failure part-way). All of these run without the network, against recorded API responses.
- The installer's icon and wizard images. Use the existing app icon.

## Open (bring back to the user only if the builder hits it)

- None at the time of writing.
