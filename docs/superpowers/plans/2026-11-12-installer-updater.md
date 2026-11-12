# Windows installer and in-app updater — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a per-user Windows installer and a portable zip on every GitHub release, built by Actions from a tag. The app updates itself, or moves to any listed version, from a Version panel.

**Architecture:**
- **One data-location module** decides dev, portable or installed mode, and every folder lookup goes through it.
- **Pure release logic** (no network): GitHub release parsing, channels, the manifest, checksum and signature checks, downgrade warnings. It lives in `crates/app/src/update/` and is tested against recorded JSON.
- **A thin network and hand-off layer** downloads off the UI thread, verifies, saves the session, runs the installer silently and exits.
- **The Version panel and toolbar chip** emit `ui::Action`s.
- **Inno Setup** builds the installer. **`.github/workflows/release.yml`** builds and attaches everything to a draft release.

**Tech Stack:** Rust (app crate), egui 0.36, winit 0.30. New plumbing crates:
- `ureq` (HTTPS, rustls)
- `serde` / `serde_json`
- `semver`
- `sha2`
- `minisign-verify`

Inno Setup 6. GitHub Actions `windows-latest`.

**Spec:** `docs/superpowers/specs/2026-11-12-installer-updater-design.md`. Read it all. Its "Builder-check rulings" section overrides the text above it: the per-user root is `%LOCALAPPDATA%\atomic-sandbox`.

## Global Constraints

- **Windows only.** Keep the existing `cfg(windows)` fences, and add no Linux or macOS packaging.
- **Per-user root `%LOCALAPPDATA%\atomic-sandbox`.** The existing `current.profile`, `timings.txt`, `training\` and `cache\pseudo` stay where they are.
  - An installed build adds `models\` (with `models\collections`), `profiles\`, `scenes\` and `captures\` under the root.
- **The dev workflow must not change.** `cargo run -p app --release`, `start.bat`, the tests and the replay scripts keep using the repo's `models/`, `profiles/` and `scenes/` exactly as now.
- **Network.** HTTPS GET only:
  - `https://api.github.com/repos/ItsKodas/atomic-sandbox/releases`, unauthenticated;
  - that repository's release asset URLs (`https://github.com/ItsKodas/atomic-sandbox/releases/download/<tag>/<name>`), and the redirect GitHub answers them with to its own asset CDN.
    - **Amended 2026-11-13** (Task 3 review I1): GitHub serves every asset from its CDN.
    - Redirects are followed by hand: at most 2 hops, HTTPS only.
    - Only these exact hosts are allowed: `api.github.com`, `github.com`, `release-assets.githubusercontent.com`, `objects.githubusercontent.com`.
    - Any other host is refused ("download redirected to an unexpected host: <host>").

  No other host, no telemetry, nothing on the UI thread. Offline shows "couldn't check (offline)" quietly, never a dialog.
- **Verification fails closed.** Never install an asset unless:
  - its SHA-256 matches `SHA256SUMS.txt`;
  - `SHA256SUMS.txt` carries a valid minisign signature (`SHA256SUMS.txt.minisig`) under the compiled-in public key.

  A build without a real public key installs nothing and says "this build has no release key".
- **Tags.** `vX.Y.Z` is a release and `vX.Y.Z-beta.N` a pre-release. The tag must equal the root `Cargo.toml` workspace `version`.
- **Never delete, move or convert user data.** Migration copies. A downgrade only warns. "Also remove my data" (unticked by default) deletes only the installed build's `models\`, `profiles\`, `scenes\` and `captures\` under the root.
- **Never interrupt a solve, a refit or a sampling pass.** Never restart without the user's click.
- **`ui::Action`.** Check now, install a version and restart are `ui::Action`s, so `--record` sees them. The channel (Stable/Beta) and the auto-check toggle are display settings stored in the profile.
- **Check interval:** at launch, then every 6 hours.
- **Publishing.** Claude never pushes a tag and never publishes a release without asking the user. Claude never handles the signing secret: the user creates and adds it.
- **Shipping.** Personal profiles and untracked models (`bh3.model`, `nh4+.model`, `ph3.model`) never ship and are never committed. Releases contain committed files only.
- **Repo rules:**
  - edit files with Edit/Write or Python, never PowerShell `Set-Content`;
  - commit with explicit paths, ending `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`;
  - keep README.md current;
  - run one heavy cargo job at a time; never kill the user's `app.exe`.
- **Tests for this work:** `cargo test -p app --release` (about 10 min). The update logic's unit tests run offline, against JSON fixtures in `crates/app/src/update/fixtures/`.

## Review Focus

1. **GitHub refuses or limits the request** (HTTP 403 or 429, or a non-JSON body). The panel shows "couldn't check (GitHub said: rate limited, try later)" and keeps the last list; nothing crashes. → Task 2 test `a_rate_limited_response_is_a_quiet_message`.
2. **A download that stops part-way, or a full disk.** The partial file is never used: download to `*.part`, verify, then rename. A retry starts clean. → Task 3 test `a_truncated_download_is_rejected_and_removed`.
3. **A Windows user name with spaces or non-ASCII letters** in `%LOCALAPPDATA%`. Paths are handled as `PathBuf` throughout, and the installer's command line is quoted. → Task 1 test `data_root_handles_spaces_and_non_ascii`, and Task 3 test `installer_command_line_quotes_paths`.
4. **A second running copy of the app at restart-to-update.** The installer must not overwrite a locked exe. The hand-off waits for the app's own exit only; Inno's `CloseApplications=yes` closes or reports the other copy, and the panel says that another window may need closing. → Task 5 script setting, and Task 4 note text test. **Amended (Task 4 fix round, controller ruling):** Restart now refuses while another copy of the install runs (`app::instance`), and the script has `CloseApplications=no`, so the installer never closes an app; it waits for `/WAITPID` up to 120 s, then calls the install off.
5. **Old releases with no `SHA256SUMS.txt` or manifest** (`v0.1.0`). They are listed with "can't be installed from the app (no signature) — download it from GitHub", plus a link. They never install unverified. → Task 2 test `an_unsigned_release_is_listed_but_not_installable`.

---

### Task 1: Data locations and the first-run import (Opus)

**Files:**
- Create `crates/app/src/paths.rs`. Add `mod paths;` in `main.rs`.
- Modify `crates/app/src/scene.rs:343-380`: `models_dir`, `collections_dir`, `scenes_dir`, `profiles_dir` and `timings_dir` delegate to `paths`.
- Modify `crates/app/src/capture/mod.rs:339-346` (`captures_dir`).
- Modify `crates/app/src/main.rs:531-537`: the root comes from `paths::user_root()`.
- Modify `crates/providers/src/pseudo.rs:621`: unchanged behaviour. Check it matches `user_root()/cache/pseudo`.
- Add a File menu item, **Import data from a folder…** (`crates/app/src/ui/menus.rs`), emitting a new `Action::ImportData(PathBuf)` in `crates/app/src/ui/actions.rs`, with record and replay support like the other actions.
- Tests in `crates/app/src/paths.rs` (`#[cfg(test)]`).

**Produces:**
- `pub enum DataMode { Dev, Portable, Installed }`
- `pub fn mode_for(exe_dir: &Path, cwd: &Path) -> DataMode`. The rule:
  1. `exe_dir/portable.txt` exists → `Portable`.
  2. `exe_dir/installed.txt` exists → `Installed`.
  3. Otherwise → `Dev`, with today's beside-exe-then-cwd lookups.
- `pub fn mode() -> DataMode`, cached.
- `pub fn user_root() -> Option<PathBuf>`: `%LOCALAPPDATA%\atomic-sandbox`.
- `pub fn dir(kind: DataDir) -> Option<PathBuf>`, with `DataDir::{Models, Collections, Scenes, Profiles, Timings, Captures}`.
  - `Installed`: under `user_root()`, created on first use.
  - `Dev` and `Portable`: exactly today's results.
- `pub fn bundled_dir(kind: DataDir) -> Option<PathBuf>`: `exe_dir/bundled/{models,scenes}` in `Installed` mode, otherwise `None`.
- `pub fn copy_tree_new(src: &Path, dst: &Path) -> Result<CopyReport, String>`.
  - It copies files that do not already exist in `dst` and never overwrites.
  - It copies into `*.part`, then renames, so nothing is half-written.
  - `CopyReport { copied: usize, skipped_existing: usize, failed: Vec<(PathBuf, String)> }`.
- `pub fn seed_installed()`.
  - On the first `Installed` launch, it copies `bundled/models` into `user_root/models` and `bundled/scenes` into `user_root/scenes`, using `copy_tree_new`.
  - It writes `user_root/installed_seeded.txt` so it runs once.
- `Action::ImportData(PathBuf)`.
  - It copies `<folder>/{models,profiles,scenes}` into the installed root with `copy_tree_new`.
  - It writes `user_root/migrated_from.txt`, containing the source path.
  - It reports the counts and any failures on the status line.
  - It is offered only in `Installed` mode. Outside that mode the menu item is disabled, with the tooltip "only the installed app keeps its data in %LOCALAPPDATA%".

**Steps:**
- [ ] Write the tests, using temp dirs and passing paths (no env, no exe):
  - `dev_mode_when_no_marker`;
  - `portable_marker_wins`;
  - `installed_marker_uses_user_root`;
  - `dev_lookups_unchanged` (the beside-exe `models/` is preferred to the cwd one, as today);
  - `data_root_handles_spaces_and_non_ascii` (a root like `C:\Users\Zoë Q\AppData\Local`);
  - `copy_tree_new_never_overwrites`;
  - `copy_tree_new_reports_a_failure_and_leaves_no_part_file`;
  - `seed_runs_once`.
- [ ] Run `cargo test -p app --release --bin app -- paths::`. Expected: FAIL, because the module is missing.
- [ ] Implement `paths.rs` and switch the call sites. `mode()` reads `std::env::current_exe()` and `current_dir()` once.
- [ ] Run the paths tests. Expected: PASS. Then run `cargo test -p app --release` in full. Expected: everything passes, with the replay scripts unchanged.
- [ ] Add `Action::ImportData` and the menu item, with an app test that the action's record line round-trips.
- [ ] Update `docs/dev-reference.md` "Build, run and the app's defaults" with the three modes. Commit.

### Task 2: Release logic — releases, channels, manifest, verification, downgrade warnings (Opus)

**Files:**
- Create `crates/app/src/update/mod.rs`, `release.rs`, `manifest.rs`, `verify.rs` and `formats.rs`.
- Create `crates/app/src/update/fixtures/`:
  - `releases.json`: a recorded API response with `v0.1.0`, a `v0.2.0-beta.1` pre-release and a `v0.2.0`;
  - `rate_limited.json`;
  - `manifest_v0.2.0.json`;
  - a test minisign key pair. **Test-only**, generated by the implementer with `minisign -G -W` (no password). Only its public key and fixture signatures are committed. The test secret key may be committed under `fixtures/` because it signs nothing real; name it `TEST-ONLY.key`.
- Create `crates/app/release-key.pub`: a placeholder line `# no release key yet`, until Task 6.
- Add the crates to `crates/app/Cargo.toml`: `serde` (derive), `serde_json`, `semver`, `sha2`, `minisign-verify`. Each gets a one-line plumbing comment, as the file's others have.
- Add a hidden CLI flag `--write-manifest <path>` in `main.rs`. It writes this build's manifest (without asset entries) and exits. CI uses it.

**Produces:**
- `pub enum Channel { Stable, Beta }`
- `pub struct Release`:

  ```rust
  pub struct Release {
      pub tag: String,
      pub version: semver::Version,
      pub prerelease: bool,
      pub date: String,
      pub notes: String,
      pub assets: Vec<Asset>,
  }
  pub struct Asset {
      pub name: String,
      pub url: String,
      pub size: u64,
  }
  ```

- `pub fn parse_releases(json: &str) -> Result<Vec<Release>, CheckError>`
  - It skips drafts and returns newest first.
  - `CheckError::{Offline, RateLimited, Http(u16), Malformed(String)}` each has a `Display` that is the panel's quiet message.
- `pub fn visible(releases: &[Release], ch: Channel) -> Vec<&Release>`: `Stable` hides pre-releases.
- `pub fn newer(releases: &[Release], ch: Channel, current: &semver::Version) -> Option<&Release>`
- `pub fn installable(r: &Release) -> Result<(&Asset, &Asset, &Asset), String>`
  - It returns the installer `AtomicSandbox-Setup-<ver>.exe`, `SHA256SUMS.txt` and `SHA256SUMS.txt.minisig`.
  - Otherwise it returns `Err("can't be installed from the app (no signature) — download it from GitHub")`.
- `pub struct Manifest { version: String, channel: String, assets: Vec<ManifestAsset>, formats: Formats }`
  - `ManifestAsset { name, size, sha256 }`.
  - `Formats` is a map of format name → `u32`. `pub fn current_formats() -> Formats` reads the real constants:
    - the collection record version (`surrogate::collection`);
    - the sidecar version (`surrogate::sidecar`);
    - the history version (`surrogate::history`);
    - the replay header version (`crate::replay::HEADER` → 1);
    - the pseudo cache magic version (`providers::pseudo` `DISK_MAGIC` → 1);
    - the checkpoint version (`providers::sampling::checkpoint`);
    - `scene` and `profile`: new code-only constants `SCENE_FORMAT: u32 = 1` (in `scene.rs`) and `PROFILE_FORMAT: u32 = 1` (in `ui/profile.rs`), each with a doc comment saying when to bump it.
  - Make any private constant `pub` as needed, without changing its value.
- `pub fn verify(sums: &[u8], sig: &[u8], file_name: &str, file_bytes_sha256: &[u8; 32], pubkey: &str) -> Result<(), String>`
  - It fails closed on a bad signature, a missing entry or a mismatch.
  - With the placeholder key, it returns `Err("this build has no release key")`.
- `pub fn downgrade_warnings(target: Option<&Formats>, current: &Formats, root: &Path) -> Vec<String>`
  - **No manifest** (`None`): one generic line, "v0.1.0 records no data formats: collections, models or profiles saved by this version may not open in it".
  - **A format whose target value is lower:** for the formats with versioned files on disk (collections, model sidecars, history, checkpoints), it lists the affected folders under `root`, read from their headers. For example: "v0.1.0 can't open 2 of your collections (they were written in a newer format). They stay untouched; that version just won't load them."
  - **For scene and profile:** one line by type, for example "scenes saved since v0.3 may not open in v0.2".

**Steps:**
- [ ] Write the tests:
  - `parses_the_recorded_releases_newest_first`;
  - `stable_hides_prereleases_beta_shows_them`;
  - `newer_is_none_when_current_is_latest`;
  - `a_rate_limited_response_is_a_quiet_message` (403 body → `CheckError::RateLimited`, Display contains "rate limited");
  - `an_unsigned_release_is_listed_but_not_installable` (v0.1.0);
  - `verify_accepts_the_fixture_and_rejects_a_flipped_byte`;
  - `verify_rejects_a_bad_signature`;
  - `placeholder_key_installs_nothing`;
  - `manifest_round_trips`;
  - `current_formats_reads_the_constants` (assert the known values);
  - `downgrade_to_no_manifest_warns_generically`;
  - `downgrade_names_newer_collections` (a temp root with one collection folder at a higher record version → one line naming it);
  - `same_formats_warn_nothing`.
- [ ] Run `cargo test -p app --release --bin app -- update::`. Expected: FAIL.
- [ ] Implement. Then run the same command. Expected: PASS.
- [ ] Add `--write-manifest`, with a test that it writes valid JSON carrying `current_formats()`. Commit.

### Task 3: Download and hand-off (Opus)

**Files:**
- Create `crates/app/src/update/net.rs`: the `ureq` client with a user agent `atomic-sandbox-updater/<version>` (no personal data). Add `ureq` with rustls to `Cargo.toml`.
- Create `crates/app/src/update/apply.rs`: the download job, the session save and the installer launch.
- Modify `crates/app/src/main.rs`: a hidden flag `--resume-session <scene file>`, which loads the scene paused, plus the profile as usual.
- Session file: `user_root/updates/session.scene`, written with the scene writer the app already uses for Save scene, with positions, velocities, MD time, charge, spin, backend, quality, ionic and walls.

**Produces:**
- `pub fn fetch_releases() -> Result<Vec<Release>, CheckError>`: blocking. It is called only from a worker thread.
- `pub struct Download`:
  - it runs on its own thread;
  - it writes `user_root/updates/<asset>.part`;
  - it reports progress `(bytes, total)` over a channel;
  - when the bytes arrive it computes SHA-256, calls `verify`, then renames to `<asset>`;
  - on failure it deletes the `.part` and reports the reason.
  - API: `pub fn start(release: &Release) -> Result<Download, String>` and `pub fn poll(&self) -> DownloadState`, where `DownloadState::{Running(u64, u64), Ready(PathBuf), Failed(String)}`.
- `pub fn installer_command(installer: &Path, exe_after: &Path, session: &Path) -> std::process::Command`
  - Arguments: `/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /RELAUNCH="<exe_after> --resume-session <session>"`. Each path is quoted. `/RELAUNCH` is a custom parameter that the Task 5 script reads.
- `pub fn hand_off(installer: &Path, session: &Path) -> Result<(), String>`
  - It spawns the installer detached, then asks the app to exit cleanly through the existing close path.
  - In `Portable` mode it is never called. Portable shows a "Download zip" link instead.

**Steps:**
- [ ] Write the tests (offline; feed `Download` from a local `std::io::Read` through an injectable source):
  - `a_truncated_download_is_rejected_and_removed`;
  - `a_verified_download_is_renamed_from_part`;
  - `installer_command_line_quotes_paths` (paths with spaces);
  - `session_scene_round_trips` (write, then load: the same positions, velocities, settings, and paused).
- [ ] Run `cargo test -p app --release --bin app -- update::apply`. Expected: FAIL.
- [ ] Implement. Then run the same command. Expected: PASS. Commit.

### Task 4: Version panel, toolbar chip and restart guard (Opus)

**Files:**
- Create `crates/app/src/ui/version.rs`: the panel. It is a window opened from **Help → Version…** (`menus.rs` `help_menu`) and by clicking the chip.
- Modify the toolbar, where the ⚠ Model only chip lives (grep `ModelOnly` in `crates/app/src/ui/`), to add the right-aligned chip **"v0.2.0 available"**.
- `crates/app/src/ui/actions.rs`: add the actions below, with record and replay lines.
- `crates/app/src/ui/profile.rs`: persist `update_channel: Channel` and `update_auto_check: bool` (default `true`).
- A worker-side or app-side update state: last check, list, error, download state. The background check runs at launch, then every 6 hours, on its own thread. It never blocks a frame.

**Produces:**
- `Action::CheckUpdates`
- `Action::InstallVersion(String)` (the tag)
- `Action::RestartToUpdate`
- `Action::CancelUpdateDownload`

**The panel, as spec "Updates during a session":**
- **Channel:** Stable / Beta.
- **Versions:** newest first, each with date and notes (collapsed), "installed" on the current one.
- **Install and restart** per entry:
  - it shows `downgrade_warnings` first, for an older target;
  - it is disabled with `installable()`'s reason for unsigned releases.
- **Progress**, then **Restart now**.
- **Check for updates now** and **Check automatically**.
- **The quiet error line.**
- **In `Portable` mode:** "Download zip" links instead of install buttons.
- **In `Dev` mode:** list only, with "running from source: updates are off".

**The restart guard:** `RestartToUpdate` while a refit, a sampling pass or a solve is running shows "Waiting for the refit to finish…" with **Cancel it and restart** and **Keep waiting**.
- With an unsaved in-session model and no collection attached, it first shows **Save model…** / **Restart anyway** / **Cancel**.
- It always mentions: "If another Atomic Sandbox window is open, the installer will ask to close it."

**Steps:**
- [ ] Write the app UI tests, in the style of `crates/app/src/ui/tests.rs`:
  - `chip_shows_only_when_newer_on_channel`;
  - `version_panel_lists_newest_first_and_marks_installed`;
  - `beta_channel_shows_prerelease`;
  - `unsigned_release_button_disabled_with_reason`;
  - `install_older_shows_downgrade_warning_first`;
  - `restart_waits_for_running_refit`;
  - `unsaved_model_offers_save_first`;
  - `portable_mode_shows_download_links`;
  - `dev_mode_says_updates_off`;
  - `buttons_emit_actions` (all four actions);
  - `channel_and_auto_check_persist_in_profile`.
- [ ] Run `cargo test -p app --release --bin app -- ui::`. Expected: the new tests FAIL.
- [ ] Implement. Then run `cargo test -p app --release` in full. Expected: PASS, with the replay scripts unchanged.
- [ ] README: add the Version panel to the controls section. Commit.

### Task 5: Inno Setup installer and committed-only packaging (Opus)

**Files:**
- Create `installer/AtomicSandbox.iss`, with:
  - `PrivilegesRequired=lowest`, `DefaultDirName={localappdata}\Programs\Atomic Sandbox`, `DisableDirPage=yes`, `DisableProgramGroupPage=yes`;
  - ~~`CloseApplications=yes`~~ **`CloseApplications=no`** (controller ruling, Task 4 fix round: the app refuses the hand-off while another copy runs, so the installer never closes an app);
  - `AppId` set to a fixed GUID, generated once and written in the script;
  - `SetupIconFile` from the generated `.ico` (copy `crates/app/build.rs`'s output, or write it with a tiny `--write-icon <path>` hidden flag);
  - files: `AtomicSandbox.exe`, `README.md`, `docs\images\`, `bundled\models\*.model` (committed models only), `bundled\scenes\*.scene`, and `installed.txt` (contents: `installed by AtomicSandbox-Setup-<ver>`);
  - a Start-menu icon;
  - a Finish-page checkbox, ticked, "Launch Atomic Sandbox";
  - a `/RELAUNCH="…"` handler that runs that command line after a silent install;
  - an uninstall page checkbox, unticked: "Also remove my models, profiles, scenes and captures (in %LOCALAPPDATA%\atomic-sandbox). Your pseudopotential cache and training checkpoints are kept." When ticked, it deletes only those four folders.
- Modify `build.bat`:
  - copy `models\` and `profiles\` from `git ls-files` only, as it already does for `scenes\`;
  - lay out `bundled\` for the installer staging folder;
  - if `ISCC.exe` is found, also build `dist\AtomicSandbox-Setup-<ver>.exe`.
  - The portable zip keeps today's layout, plus `portable.txt`.

**Steps:**
- [ ] Write the script and the `build.bat` changes.
- [ ] Verify by hand, and record it in the report:
  - run `build.bat`;
  - check the staging folder lists only committed models (no `bh3.model`, `nh4+.model` or `ph3.model`) and no personal profiles;
  - if Inno Setup is installed locally, build the installer and run a silent install into a temp `/DIR=`.
  - **Do not install over the user's environment, and never touch `%LOCALAPPDATA%\atomic-sandbox`.** If Inno isn't installed, leave that check to CI (Task 6) and say so.
- [ ] Commit.

### Task 6: Release workflow, signing key and docs (Opus)

**Files:**
- Create `.github/workflows/release.yml`.
  - **Trigger:** `on: push: tags: ['v*']`, runner `windows-latest`.
  - **The tag check:** the tag (without `v`) must equal the root `Cargo.toml` `version`, or fail.
  - **Build and package:** `cargo build -p app --release`. Run the `build.bat` layout steps non-interactively (factor them into `scripts/package.ps1` or a `build.bat --ci` flag, so CI and local share one layout). Then ISCC (preinstalled on `windows-latest`; if not, install `innosetup` via choco).
  - **Checksums and signature:**
    - write `SHA256SUMS.txt` over the installer and the zip;
    - sign it with minisign, using the secrets `MINISIGN_SECRET_KEY` and `MINISIGN_PASSWORD` (install `minisign` via choco or a pinned release download);
    - run `AtomicSandbox.exe --write-manifest manifest.json`, then fill in the assets' name, size and sha256.
  - **The release:** `gh release create <tag> --draft --title "Atomic Sandbox <ver>" --notes-file <notes>`, with `--prerelease` when the tag contains `-beta.`. Attach the installer, the zip, `SHA256SUMS.txt`, `SHA256SUMS.txt.minisig` and `manifest.json`. The notes include the line: "Windows may show 'Windows protected your PC' — the installer isn't code-signed yet: More info → Run anyway."
  - **Permissions:** `contents: write` only.
- `crates/app/release-key.pub`: **left as the placeholder.** The controller gives the user these commands, and the user runs them:

  ```bash
  minisign -G -p release-key.pub -s release-key.sec
  ```

  ```bash
  gh secret set MINISIGN_SECRET_KEY < release-key.sec
  ```

  ```bash
  gh secret set MINISIGN_PASSWORD
  ```

  Then the user, or Claude on their word, commits the `.pub` contents into `crates/app/release-key.pub`. The `.sec` never enters the repo.
- `docs/dev-reference.md`: a new section, "Releasing":
  - the tag scheme;
  - the checklist: run the gates locally on the tagged commit (`cargo test -p units -p qm -p sim -p surrogate --release`, `cargo test -p providers --release`, `cargo test -p validate --release`, `cargo test -p app --release`), push the tag (ask first), check the draft, publish (ask first);
  - when to bump `SCENE_FORMAT` and `PROFILE_FORMAT`.
- `README.md`: rewrite the install section with the installer first (and the SmartScreen line), the portable zip second, and where data lives.
- `CLAUDE.md`: one line under "Build and run" pointing to dev-reference "Releasing".

**Steps:**
- [ ] Write the workflow, and validate its syntax with `actionlint` if available. Otherwise review it carefully, and say so.
- [ ] Docs. Commit.
- [ ] **Stop here.** The first real tag push is a publishing step. The controller asks the user before any `git push --tags`, and only after the user has added the secrets and the public key is committed.

---

## Self-review notes

- **Spec coverage:**
  - Goal 1 (installer): Task 5.
  - Goal 2 (zip): Task 5.
  - Goal 3 (data and migration): Task 1.
  - Goal 4 (updater): Tasks 2–4.
  - Goal 5 (workflow): Task 6.
  - Anti-goals: fail-closed `verify` (Task 2), copy-only (Task 1), the restart guard (Task 4), downgrade warnings (Task 2), dev unchanged (Task 1's `dev_lookups_unchanged`, plus the full app suite).
  - Builder-check rulings: the root (Task 1), uninstall scope (Task 5), scene and profile format constants (Task 2), restart contents (Tasks 3–4), committed-only release (Task 5).
- **Interfaces:** Tasks 3 and 4 use Task 2's `Release`, `Asset`, `Channel`, `CheckError`, `installable`, `verify` and `downgrade_warnings`, and Task 1's `paths::mode()` and `user_root()`, by the names above.
- **My calls**, not stated in the spec:
  - the `installed.txt` marker and the `bundled\` folder name. A `models\` beside the installed exe would otherwise make it look like dev mode.
  - unsigned old releases are listed but not installable, which follows from "fail closed";
  - the test key is committed, marked `TEST-ONLY`.
