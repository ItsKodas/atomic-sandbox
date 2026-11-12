//! Help ▸ Version… and the top bar's "v0.2.0 available" chip (installer spec, "Updates during a
//! session" and "Downgrades"; plan Task 4). No popups, ever: the chip appears at the top bar's
//! right end when a newer version is out on the chosen channel, and clicking it opens this panel.
//!
//! **The panel** lists every release of the channel (Stable: releases only; Beta: pre-releases
//! too), newest first, each with its date and its release notes (collapsed), the running one
//! marked "installed". What an entry offers depends on the build (`crate::paths::DataMode`):
//! - **Installed:** Install and restart. The download runs in the background (`update::Download`:
//!   verified, then renamed into place) and the panel shows its progress; when it is ready the
//!   button becomes **Restart now**. A release that can't be installed from the app (no signature,
//!   no installer: `update::installable`) has the button disabled with the reason. An **older**
//!   target first shows its downgrade warnings (`update::downgrade_warnings`, against the
//!   release's `manifest.json`, fetched small into memory when the release has one); only
//!   "Install … anyway" then starts it.
//! - **Portable:** a "Download zip" link instead (the zip is this repository's own release asset).
//!   No downgrade warnings: a portable copy installs nothing.
//! - **Dev:** the list only — "running from source: updates are off".
//!
//! **One download at a time.** Starting another cancels the first; a cancel takes effect at the
//! job's next read, up to the 30 s read timeout on a stalled connection, and the panel says
//! "Cancelling…" until the job has reported.
//!
//! **The check** runs on its own thread, never in a frame: Check for updates now (any build), and
//! — in the window of an installed or portable build with Check automatically on — at launch and
//! then again once [`CHECK_EVERY`] has passed since the last one started. That is tested against
//! the last check's `Instant` at every frame, so a window that draws no frame (idle, minimised)
//! checks at its next frame after the 6 hours, not on the minute. A build from source never checks
//! by itself, so the tests, replays and `start.bat` make no network call. A failed check is one
//! quiet line, and the last list stays.
//!
//! **The restart guard** (`Action::RestartToUpdate`; [`Guard`]):
//! 1. A model the restart wouldn't keep — samples learned since it was last loaded from or saved to
//!    a file, with no shared collection attached ([`save_note`]) — first gets Save model… /
//!    Restart anyway / Cancel, saying what would be lost.
//! 2. A playing simulation is never paused by itself: "The simulation is playing." with **Pause and
//!    restart** (the pause, as the Pause button sends it — the step being solved is dropped — then
//!    the restart once the run has stopped) / Cancel. Play pressed again while it pauses (once the
//!    pause has been seen to land) goes back to that question, so a later plain Pause never
//!    restarts the app.
//! 3. A background job — a sampling pass, a refit, a relax, a background audit ([`running`]) — or,
//!    with the run paused, a solve the worker is still busy with (the one after a scene load, an
//!    Apply grid or a Step; Bonds now; `ui::auto_bonds`' refresh after a drag: [`waiting_for`])
//!    gets "Waiting for the refit to finish…" with Cancel it and restart / Keep waiting; the restart
//!    then goes ahead by itself when the job ends.
//!
//! Closing the Version window calls a pending restart off (its card is drawn only there): the
//! status line says so, and the download stays ready. Every way back to [`Guard::Idle`] clears the
//! guard's flags. The restart itself is the viewer's (`PanelOutput::restart_for_update`): refused
//! while another copy of this install runs (`crate::instance` — the silent installer must not
//! close it), else the session saved, the verified installer handed off, the app closed through its
//! own exit path.
//!
//! The channel and Check automatically are display settings kept in the profile; Check now,
//! Install, Restart now and Cancel are `ui::Action`s, so `--record` sees them (a replay skips them:
//! it never touches the network or installs).
use super::{probed, theme, Action, PanelInput, PanelOutput, UiState, AMBER, GREEN};
use crate::paths::DataMode;
use crate::update::release::MANIFEST_NAME;
use crate::update::{self, Channel, CheckError, Download, DownloadState, Formats, Release, Verified};
use egui::{Button, Margin, RichText, Stroke, Ui};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The Dev build's line (and its refusal of an install).
pub const UPDATES_OFF: &str = "running from source: updates are off";
/// A portable copy's refusal of an install.
pub const PORTABLE_REFUSED: &str = "a portable copy isn't updated in place: download the zip instead";
/// The playing card's line (ruling I2: the restart never pauses by itself).
pub const PLAYING: &str = "The simulation is playing.";
/// The status line when closing the Version window calls a pending restart off (ruling I1).
pub const CALLED_OFF: &str = "Restart for the update called off";
/// The background check's interval (after the one at launch).
pub const CHECK_EVERY: Duration = Duration::from_secs(6 * 3600);

/// What the panel asks of the network, each call blocking (they run on threads of their own).
/// The app's is [`GitHub`]; the tests stand in their own, so no test touches the network.
pub trait UpdateNet: Send + Sync {
    /// The releases list (`update::fetch_releases`).
    fn releases(&self) -> Result<Vec<Release>, CheckError>;
    /// The release's data formats from its `manifest.json`; `None` when it has none.
    fn manifest(&self, release: &Release) -> Result<Option<Formats>, String>;
    /// Starts the release's verified download into `dir` (`<user root>\updates`).
    fn download(&self, release: &Release, dir: PathBuf) -> Result<Download, String>;
}

/// GitHub, over `update::net` (HTTPS GET, GitHub's hosts only).
#[cfg_attr(test, allow(dead_code))]
pub struct GitHub;

impl UpdateNet for GitHub {
    fn releases(&self) -> Result<Vec<Release>, CheckError> {
        update::fetch_releases()
    }

    fn manifest(&self, release: &Release) -> Result<Option<Formats>, String> {
        update::fetch_manifest(release).map(|m| m.map(|m| m.formats))
    }

    fn download(&self, release: &Release, dir: PathBuf) -> Result<Download, String> {
        let source = Box::new(update::net::Http::new());
        #[cfg(not(test))]
        return Download::start_with(release, dir, source);
        #[cfg(test)]
        return Download::start_with(release, dir, source, update::verify::RELEASE_KEY);
    }
}

/// The tests' default network: nothing reachable, nothing sent.
#[cfg(test)]
pub struct Offline;

#[cfg(test)]
impl UpdateNet for Offline {
    fn releases(&self) -> Result<Vec<Release>, CheckError> {
        Err(CheckError::Offline)
    }
    fn manifest(&self, _: &Release) -> Result<Option<Formats>, String> {
        Err("offline (test)".into())
    }
    fn download(&self, _: &Release, _: PathBuf) -> Result<Download, String> {
        Err("offline (test)".into())
    }
}

/// The restart guard's state (module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Guard {
    Idle,
    /// A model the restart wouldn't keep ([`save_note`]): Save model… / Restart anyway / Cancel.
    SaveFirst,
    /// The simulation is playing: Pause and restart / Cancel.
    Playing,
    /// Pause and restart was pressed: the restart follows once the run has stopped. `landed`: a
    /// frame has seen the run paused since (until then the pause is still on its way); playing
    /// again after that is the user's Play, which goes back to [`Guard::Playing`].
    Pausing { landed: bool },
    /// A background job, or a paused run's solve, runs: the restart waits for it
    /// ([`waiting_for`]). `prompt`: the card with Cancel it and
    /// restart / Keep waiting (Keep waiting leaves the quieter "Restarting when … finishes" line).
    Waiting { what: &'static str, prompt: bool },
}

/// Installing an older version: its downgrade warnings, shown before anything starts.
pub struct Confirm {
    pub tag: String,
    version: semver::Version,
    /// `None` while the release's manifest is being read.
    pub warnings: Option<Vec<String>>,
    rx: Option<Receiver<Result<Option<Formats>, String>>>,
}

/// The updater's state: the list, the check, the download, the guard (module doc). Display
/// state, apart from what the actions change.
pub struct UpdateState {
    /// The Version panel is open.
    pub open: bool,
    /// Stable or Beta (the profile's `update_channel`).
    pub channel: Channel,
    /// Check at launch and every 6 hours (the profile's `update_auto_check`; on by default).
    pub auto_check: bool,
    /// The window's own state (`main`): the automatic checks may run. Off headless (screenshots,
    /// replays, tests), so they never reach the network by themselves.
    pub background: bool,
    /// This build's version (`CARGO_PKG_VERSION`; a test stands in another).
    pub current: semver::Version,
    /// The last list a check returned, newest first (kept when a later check fails).
    pub releases: Vec<Release>,
    /// The last check's quiet error line.
    pub error: Option<String>,
    /// When the last check ended, and when the last one started (the 6-hour test, every frame).
    pub checked_at: Option<Instant>,
    pub last_started: Option<Instant>,
    check: Option<Receiver<Result<Vec<Release>, CheckError>>>,
    /// The one download (module doc), and the cancelled ones still finishing a read.
    pub download: Option<Download>,
    cancelling: Vec<Download>,
    /// An older target's warnings, waiting for "Install … anyway".
    pub confirm: Option<Confirm>,
    pub guard: Guard,
    /// The guard's "Restart anyway" (the model) and "Cancel it and restart" (the job) were
    /// pressed: the next `RestartToUpdate` skips that question. Cleared on every way back to Idle.
    pub model_ack: bool,
    pub force: bool,
    /// What the SaveFirst card says would be lost ([`save_note`], set as it opens).
    pub save_text: String,
    /// The model's samples when it was last loaded from or saved to a file (ruling M1), and the
    /// file the model was last seen loaded from (a change of it is a load).
    pub model_baseline: Option<usize>,
    pub seen_file: Option<PathBuf>,
    pub net: Arc<dyn UpdateNet>,
}

impl Default for UpdateState {
    fn default() -> Self {
        #[cfg(not(test))]
        let net: Arc<dyn UpdateNet> = Arc::new(GitHub);
        #[cfg(test)]
        let net: Arc<dyn UpdateNet> = Arc::new(Offline);
        UpdateState {
            open: false,
            channel: Channel::Stable,
            auto_check: true,
            background: false,
            current: semver::Version::parse(env!("CARGO_PKG_VERSION")).expect("the package version is semver"),
            releases: Vec::new(),
            error: None,
            checked_at: None,
            last_started: None,
            check: None,
            download: None,
            cancelling: Vec::new(),
            confirm: None,
            guard: Guard::Idle,
            model_ack: false,
            force: false,
            save_text: String::new(),
            model_baseline: None,
            seen_file: None,
            net,
        }
    }
}

impl UpdateState {
    /// A check is in flight.
    pub fn checking(&self) -> bool {
        self.check.is_some()
    }

    /// A cancelled download hasn't reported yet ("Cancelling…").
    pub fn cancelling(&self) -> bool {
        !self.cancelling.is_empty()
    }

    /// The releases the channel lists, newest first.
    pub fn listed(&self) -> Vec<&Release> {
        update::visible(&self.releases, self.channel)
    }

    /// The newest release on the channel newer than this build: what the chip offers.
    pub fn newer(&self) -> Option<&Release> {
        update::newer(&self.releases, self.channel, &self.current)
    }

    /// The downloaded, verified installer, once ready (`Restart now`).
    pub fn verified(&self) -> Option<Verified> {
        self.download.as_ref()?.verified()
    }

    /// The download's state, when it is `tag`'s.
    fn state_of(&self, tag: &str) -> Option<DownloadState> {
        self.download.as_ref().filter(|d| d.tag() == tag).map(Download::poll)
    }

    /// No restart pending: the guard Idle and its flags cleared (ruling I4: every way back).
    pub fn idle(&mut self) {
        self.guard = Guard::Idle;
        (self.model_ack, self.force) = (false, false);
    }

    /// The model was saved to a file now (`Action::SaveModel`): what it holds is the baseline.
    pub fn note_saved(&mut self, input: &PanelInput<'_>) {
        if let Some(m) = input.model() {
            self.model_baseline = Some(m.samples);
        }
    }

    fn start_check(&mut self, now: Instant) {
        let net = self.net.clone();
        let (tx, rx) = channel();
        self.last_started = Some(now);
        match std::thread::Builder::new().name("update-check".into()).spawn(move || {
            let _ = tx.send(net.releases());
        }) {
            Ok(_) => self.check = Some(rx),
            Err(e) => self.error = Some(format!("couldn't check (the check couldn't start: {e})")),
        }
    }
}

/// A failed check's quiet line: `CheckError`'s own words, except that a redirect the client
/// refused (`update::net`, a `CheckError::Malformed` carrying why) is said as it is, not as an
/// unreadable answer.
pub fn check_error_line(e: &CheckError) -> String {
    match e {
        CheckError::Malformed(why) if why.starts_with("GitHub's releases list redirected") => format!("couldn't check: {why}"),
        e => e.to_string(),
    }
}

/// The chip's text, "v0.2.0 available", when a newer version is out on the channel.
pub fn chip_text(u: &UpdateState) -> Option<String> {
    u.newer().map(|r| format!("{} available", r.tag))
}

/// The release's zip ("Download zip" in a portable copy): its `.zip` asset when that is exactly
/// one of this release's own asset URLs.
pub fn zip_link(r: &Release) -> Option<&str> {
    r.assets.iter().find(|a| a.name.ends_with(".zip") && update::net::check_asset(&r.tag, a).is_ok()).map(|a| a.url.as_str())
}

/// The background job the restart waits for (module doc), named: a sampling pass, a refit, a
/// relax, a background audit; `None` when none runs. The step solves of a playing run are not one:
/// a playing run gets its own question ([`Guard::Playing`]).
pub fn running(input: &PanelInput<'_>) -> Option<&'static str> {
    let probe = input.probe.as_ref();
    if input.sampling() {
        Some("the sampling pass")
    } else if probe.is_some_and(|p| p.refit.is_some()) {
        Some("the refit")
    } else if input.relaxing() {
        Some("the relax")
    } else if probe.is_some_and(|p| p.side.running) {
        Some("the background audit")
    } else {
        None
    }
}

/// What the restart waits for (module doc): a background job ([`running`]), else — the run paused —
/// a solve the worker is still busy with, "the solve" (the one after a scene load, an Apply grid or
/// a Step; Bonds now; `ui::auto_bonds`' refresh after a drag). A playing run's step solves are not
/// one: a playing run gets [`Guard::Playing`].
pub fn waiting_for(input: &PanelInput<'_>) -> Option<&'static str> {
    running(input).or_else(|| (input.busy && !input.playing).then_some("the solve"))
}

/// The playing run's question; a "Cancel it and restart" given earlier is spent (it was for the
/// job, not for whatever runs after the next pause).
fn ask_playing(u: &mut UpdateState) {
    u.guard = Guard::Playing;
    u.force = false;
}

/// What the restart would lose of the estimated solver's model, in words, or `None` when it keeps
/// it all (ruling M1): nothing is lost from a shared collection (on disk sample by sample, and
/// re-attached after the restart), nor what a model file already holds — only the samples learned
/// since the model was last loaded from or saved to a file (`UpdateState::model_baseline`), or
/// all of a model this session learned and never saved.
pub fn save_note(u: &UpdateState, input: &PanelInput<'_>) -> Option<String> {
    let m = input.model()?;
    if m.samples == 0 || matches!(m.source, sim::ModelSource::Collection { .. }) {
        return None;
    }
    let since = u.model_baseline.filter(|&b| b <= m.samples);
    let new = m.samples - since.unwrap_or(0);
    if new == 0 {
        return None;
    }
    let s = |n: usize| if n == 1 { "" } else { "s" };
    Some(match (&m.source, since) {
        (sim::ModelSource::File(p), Some(_)) if !m.source_is_autosave => format!(
            "The model from {} has learned {new} sample{} since it was loaded or saved. The restart reopens the scene without them (the file itself stays as it was): save the model first to keep them.",
            p.file_name().map_or_else(|| p.display().to_string(), |n| n.to_string_lossy().into_owned()),
            s(new)
        ),
        (_, Some(_)) => format!("This session's model has learned {new} sample{} since you last saved it. The restart doesn't keep them: save the model first to keep them.", s(new)),
        (_, None) => format!("This session's model ({new} sample{}) isn't in a shared collection or a saved file, so the restart doesn't keep it: save it first to keep it.", s(new)),
    })
}

/// The installed build's refusals of an update (the other two modes).
fn refused(mode: DataMode) -> Option<&'static str> {
    match mode {
        DataMode::Installed => None,
        DataMode::Portable => Some(PORTABLE_REFUSED),
        DataMode::Dev => Some(UPDATES_OFF),
    }
}

// ---- the actions ----

/// `Action::CheckUpdates`: a check on its own thread (none while one runs).
pub(super) fn check_now(st: &mut UiState) {
    if !st.update.checking() {
        st.update.start_check(Instant::now());
    }
}

/// `Action::InstallVersion(tag)`: the release's download, verified in the background (module
/// doc). Starting it cancels another, and any pending restart; the same release already
/// downloading or ready is left.
pub(super) fn install(st: &mut UiState, tag: &str, out: &mut PanelOutput) {
    if let Some(why) = refused(st.data_mode) {
        out.error = Some(why.into());
        return;
    }
    let Some(r) = st.update.releases.iter().find(|r| r.tag == tag).cloned() else {
        out.error = Some(format!("install {tag}: it isn't in the list (Check for updates now first)"));
        return;
    };
    if let Err(why) = update::installable(&r) {
        out.error = Some(format!("{tag} {why}"));
        return;
    }
    if r.version == st.update.current {
        out.error = Some(format!("{tag} is the version running"));
        return;
    }
    if st.update.state_of(tag).is_some_and(|s| !matches!(s, DownloadState::Failed(_))) {
        return;
    }
    let Some(root) = st.data_root.clone() else {
        out.error = Some("install: there is no per-user folder to download into (LOCALAPPDATA isn't set)".into());
        return;
    };
    match st.update.net.download(&r, root.join("updates")) {
        Ok(d) => {
            if let Some(old) = st.update.download.replace(d) {
                old.cancel();
                if matches!(old.poll(), DownloadState::Running(..)) {
                    st.update.cancelling.push(old);
                }
            }
            st.update.confirm = None;
            st.update.idle();
            st.set_status(format!("downloading {tag}: it is verified before anything is installed"));
        }
        Err(e) => out.error = Some(format!("install {tag}: {e}")),
    }
}

/// `Action::CancelUpdateDownload`: the download stops (a running one reports at its next read;
/// "Cancelling…" until then), and a pending restart is called off.
pub(super) fn cancel_download(st: &mut UiState) {
    st.update.idle();
    let Some(d) = st.update.download.take() else { return };
    if matches!(d.poll(), DownloadState::Running(..)) {
        d.cancel();
        st.set_status(format!("cancelling the download of {}…", d.tag()));
        st.update.cancelling.push(d);
    }
}

/// `Action::RestartToUpdate` and the guard (module doc): the restart goes ahead
/// (`PanelOutput::restart_for_update`) only with a verified installer, past the model question,
/// with the simulation not playing, and with no background job running (or "Cancel it and
/// restart"). The status line is the viewer's, once the hand-off has worked (ruling M2).
pub(super) fn restart(st: &mut UiState, input: &PanelInput<'_>, out: &mut PanelOutput) {
    if let Some(why) = refused(st.data_mode) {
        out.error = Some(why.into());
        return;
    }
    if st.update.verified().is_none() {
        out.error = Some("restart to update: nothing is downloaded and verified yet (Install and restart a version first)".into());
        return;
    }
    if !st.update.model_ack {
        if let Some(text) = save_note(&st.update, input) {
            st.update.save_text = text;
            st.update.guard = Guard::SaveFirst;
            return;
        }
    }
    if input.playing {
        ask_playing(&mut st.update);
        return;
    }
    if !st.update.force {
        if let Some(what) = waiting_for(input) {
            let prompt = match st.update.guard {
                Guard::Waiting { prompt, .. } => prompt,
                _ => true,
            };
            st.update.guard = Guard::Waiting { what, prompt };
            return;
        }
    }
    st.update.idle();
    out.restart_for_update = true;
}

/// Closing the Version window (ruling I1): a pending restart is called off — its card is drawn only
/// in the window — with a status line; the download stays ready, so Restart now is offered again.
pub fn close_panel(st: &mut UiState) {
    st.update.open = false;
    if st.update.guard != Guard::Idle {
        st.update.idle();
        st.set_status(CALLED_OFF);
    }
}

/// Once a frame, after the actions (`ui::show`): the check's answer, the automatic check, the
/// cancelled downloads, an older target's manifest, the model's load baseline, and a pending
/// restart once its run has stopped or its job is done. Never blocks.
pub(super) fn tick(ctx: &egui::Context, st: &mut UiState, input: &PanelInput<'_>, out: &mut PanelOutput) {
    let now = Instant::now();
    let u = &mut st.update;
    if let Some(rx) = &u.check {
        let got = match rx.try_recv() {
            Ok(r) => Some(r.map_err(|e| check_error_line(&e))),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(Err("couldn't check (the check stopped unexpectedly)".to_string())),
        };
        if let Some(r) = got {
            u.check = None;
            u.checked_at = Some(now);
            match r {
                Ok(list) => (u.releases, u.error) = (list, None),
                Err(e) => u.error = Some(e),
            }
        }
    }
    // the 6-hour test against the last check's start, every frame (module doc)
    let automatic = u.background && u.auto_check && st.data_mode != DataMode::Dev;
    let due = u.last_started.is_none_or(|t| now.saturating_duration_since(t) >= CHECK_EVERY);
    if automatic && due && u.check.is_none() {
        u.start_check(now);
    }
    u.cancelling.retain(|d| matches!(d.poll(), DownloadState::Running(..)));
    // a model newly loaded from a file: what it holds now is the baseline (ruling M1); a session's
    // autosave isn't the user's file (it is removed at the exit)
    if let Some(m) = input.model() {
        let file = match &m.source {
            sim::ModelSource::File(p) if !m.source_is_autosave => Some(p.clone()),
            _ => None,
        };
        if file != u.seen_file {
            u.model_baseline = file.as_ref().map(|_| m.samples);
            u.seen_file = file;
        }
        // fewer samples than the baseline: another model (forgotten and learned anew)
        if u.model_baseline.is_some_and(|b| m.samples < b) {
            u.model_baseline = None;
        }
    }
    poll_confirm(st);
    match st.update.guard {
        // (playing again with no job left: `restart` asks the playing question)
        Guard::Waiting { what, prompt } => match waiting_for(input) {
            None => restart(st, input, out),
            Some(w) if w != what => st.update.guard = Guard::Waiting { what: w, prompt },
            Some(_) => {}
        },
        // the pause has landed (the step it stopped has let go of the solver too)
        Guard::Pausing { .. } if !input.playing && !input.busy => restart(st, input, out),
        Guard::Pausing { landed: false } if !input.playing => st.update.guard = Guard::Pausing { landed: true },
        // paused, then Play again: ask again (T4 N5) — a later plain Pause must not restart
        Guard::Pausing { landed: true } if input.playing => ask_playing(&mut st.update),
        _ => {}
    }
    let u = &st.update;
    let downloading = u.download.as_ref().is_some_and(|d| matches!(d.poll(), DownloadState::Running(..)));
    let reading = u.confirm.as_ref().is_some_and(|c| c.rx.is_some());
    if u.checking() || u.cancelling() || downloading || reading || u.guard != Guard::Idle {
        ctx.request_repaint_after(Duration::from_millis(250));
    }
}

/// An older target's manifest, once read: its warnings.
fn poll_confirm(st: &mut UiState) {
    let root = st.data_root.clone();
    let Some(c) = st.update.confirm.as_mut() else { return };
    let Some(rx) = &c.rx else { return };
    let got = match rx.try_recv() {
        Ok(r) => r,
        Err(TryRecvError::Empty) => return,
        Err(TryRecvError::Disconnected) => Err("the read stopped unexpectedly".into()),
    };
    c.rx = None;
    c.warnings = Some(warnings_for(&c.version, got, root.as_deref()));
}

/// The downgrade warnings for `v` (`update::downgrade_warnings` against the user's data under
/// `root`): from its manifest's formats, or the generic line when it records none — or when its
/// manifest couldn't be read, which is said first.
fn warnings_for(v: &semver::Version, manifest: Result<Option<Formats>, String>, root: Option<&Path>) -> Vec<String> {
    let current = update::current_formats();
    // (without a target manifest the warning is the generic one, which reads no file)
    let generic = || update::downgrade_warnings(v, None, &current, Path::new(""));
    match (manifest, root) {
        (Ok(Some(f)), Some(root)) => update::downgrade_warnings(v, Some(&f), &current, root),
        (Ok(_), _) => generic(),
        (Err(e), _) => std::iter::once(format!("couldn't read v{v}'s data-format manifest ({e})")).chain(generic()).collect(),
    }
}

/// Install and restart on an older entry: its warnings first (the manifest read on a thread when
/// the release carries one).
fn ask_downgrade(st: &mut UiState, r: &Release) {
    let mut c = Confirm { tag: r.tag.clone(), version: r.version.clone(), warnings: None, rx: None };
    if r.asset(MANIFEST_NAME).is_none() {
        c.warnings = Some(warnings_for(&r.version, Ok(None), st.data_root.as_deref()));
    } else {
        let (net, release, (tx, rx)) = (st.update.net.clone(), r.clone(), channel());
        let spawned = std::thread::Builder::new().name("update-manifest".into()).spawn(move || {
            let _ = tx.send(net.manifest(&release));
        });
        match spawned {
            Ok(_) => c.rx = Some(rx),
            Err(e) => c.warnings = Some(warnings_for(&r.version, Err(e.to_string()), st.data_root.as_deref())),
        }
    }
    st.update.confirm = Some(c);
}

// ---- drawing ----

/// The top bar's chip (`menus::top_bar`, right-aligned beside the search): "v0.2.0 available"
/// while a newer version is out on the channel; a click opens the panel.
pub(super) fn chip(ui: &mut Ui, st: &mut UiState) -> Option<egui::Response> {
    let text = chip_text(&st.update)?;
    let b = Button::new(RichText::new(text).size(12.5).color(theme::ACCENT_INK)).fill(theme::ACCENT_BG).stroke(Stroke::new(1.0, theme::ACCENT)).corner_radius(egui::CornerRadius::same(12));
    let r = probed("update-chip", ui.add(b)).on_hover_text("a newer version is out on your channel: the Version panel (Help ▸ Version…) installs it when you choose");
    if r.clicked() {
        st.update.open = true;
    }
    Some(r)
}

/// The Version panel, while open. Its close button calls a pending restart off ([`close_panel`]).
pub(super) fn show(ctx: &egui::Context, st: &mut UiState, acts: &mut Vec<Action>) {
    if !st.update.open {
        return;
    }
    let mut open = true;
    egui::Window::new("Version")
        .id(egui::Id::new("version-window"))
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .default_width(580.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .show(ctx, |ui| body(ui, st, acts));
    if !open {
        close_panel(st);
    }
}

fn body(ui: &mut Ui, st: &mut UiState, acts: &mut Vec<Action>) {
    ui.set_width(580.0);
    let mode = st.data_mode;
    ui.label(RichText::new(format!("Atomic Sandbox v{}", st.update.current)).size(16.0).strong());
    match mode {
        DataMode::Installed => theme::note(ui, "The installed build: it installs any signed version from this repository's GitHub releases, verified before anything is replaced."),
        DataMode::Portable => theme::note(ui, "A portable copy: it isn't updated in place — download a release's zip to change version."),
        DataMode::Dev => ui.label(RichText::new(UPDATES_OFF).color(AMBER)),
    };
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label("Channel");
        for (ch, word, tip, name) in [(Channel::Stable, "Stable", "releases only", "channel-stable"), (Channel::Beta, "Beta", "pre-releases too", "channel-beta")] {
            if probed(name, ui.add(Button::selectable(st.update.channel == ch, word))).on_hover_text(tip).clicked() {
                st.update.channel = ch;
            }
        }
    });
    ui.horizontal(|ui| {
        let checking = st.update.checking();
        if probed("check-now", ui.add_enabled(!checking, Button::new("Check for updates now")))
            .on_hover_text("only GitHub's public release list is read (an HTTPS GET); GitHub sees your IP address as with any web request")
            .clicked()
        {
            acts.push(Action::CheckUpdates);
        }
        let r = probed("auto-check", ui.add_enabled(mode != DataMode::Dev, egui::Checkbox::new(&mut st.update.auto_check, "Check automatically")));
        r.on_hover_text("at launch, then every 6 hours while the app is open (checked at the next frame once they have passed); it only ever shows the chip, never a popup").on_disabled_hover_text(UPDATES_OFF);
        if checking {
            ui.spinner();
            ui.label(RichText::new("checking…").color(theme::INK_DIM));
        } else if let Some(t) = st.update.checked_at {
            ui.label(RichText::new(format!("checked {} ago", super::duration_label(t.elapsed().as_secs_f64()))).color(theme::INK_FAINT));
        }
    });
    if let Some(e) = &st.update.error {
        ui.label(RichText::new(e).size(12.5).color(theme::INK_DIM));
    }
    if st.update.cancelling() {
        ui.label(RichText::new("Cancelling…").color(theme::INK_DIM)).on_hover_text("the download stops at its next read: on a stalled connection that can take up to 30 s");
    }
    guard_card(ui, st, acts);
    ui.separator();
    let listed: Vec<Release> = st.update.listed().into_iter().cloned().collect();
    if listed.is_empty() {
        theme::note(ui, if st.update.checked_at.is_none() { "No list yet: Check for updates now." } else { "No releases on this channel." });
    }
    egui::ScrollArea::vertical().max_height(400.0).auto_shrink([false, true]).show(ui, |ui| {
        for r in &listed {
            entry(ui, st, r, acts);
            ui.add_space(4.0);
        }
    });
}

/// A card in the panel (the guard, the downgrade warnings).
fn card<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    egui::Frame::new()
        .fill(theme::ACCENT_BG)
        .stroke(Stroke::new(1.0, AMBER))
        .corner_radius(6)
        .inner_margin(Margin::symmetric(10, 8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui)
        })
        .inner
}

/// The restart guard's card (module doc).
fn guard_card(ui: &mut Ui, st: &mut UiState, acts: &mut Vec<Action>) {
    match st.update.guard {
        Guard::Idle => {}
        Guard::SaveFirst => card(ui, |ui| {
            ui.label(RichText::new("The restart won't keep what this model learned").strong());
            theme::note(ui, st.update.save_text.clone());
            ui.horizontal(|ui| {
                if probed("guard-save", ui.button("Save model…")).on_hover_text("save the model to a file; then Restart now again").clicked() {
                    st.update.idle();
                    super::files::open_save_model(st);
                }
                if probed("guard-anyway", ui.button("Restart anyway")).on_hover_text("restart without what it learned").clicked() {
                    st.update.model_ack = true;
                    acts.push(Action::RestartToUpdate);
                }
                if probed("guard-cancel", ui.button("Cancel")).clicked() {
                    st.update.idle();
                }
            });
        }),
        Guard::Playing => card(ui, |ui| {
            ui.label(RichText::new(PLAYING).strong());
            theme::note(ui, "The restart doesn't stop it by itself. Pause and restart pauses the run as the Pause button does (the step being solved is dropped), then restarts; the app reopens paused where it stopped.");
            ui.horizontal(|ui| {
                if probed("guard-pause-restart", ui.button("Pause and restart")).clicked() {
                    acts.push(Action::Pause);
                    st.update.guard = Guard::Pausing { landed: false };
                }
                if probed("guard-cancel", ui.button("Cancel")).clicked() {
                    st.update.idle();
                }
            });
        }),
        Guard::Pausing { .. } => card(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Pausing, then restarting…");
                if probed("guard-dont-restart", ui.button("Don't restart")).clicked() {
                    st.update.idle();
                }
            });
        }),
        Guard::Waiting { what, prompt: true } => card(ui, |ui| {
            ui.label(RichText::new(format!("Waiting for {what} to finish…")).strong());
            theme::note(ui, format!("The restart waits for it and goes ahead when it ends. Or cancel it: the app closes now and {what} stops with it, as when you close the window."));
            ui.horizontal(|ui| {
                if probed("guard-cancel-restart", ui.button("Cancel it and restart")).clicked() {
                    st.update.force = true;
                    acts.push(Action::RestartToUpdate);
                }
                if probed("guard-keep-waiting", ui.button("Keep waiting")).clicked() {
                    st.update.guard = Guard::Waiting { what, prompt: false };
                }
            });
        }),
        Guard::Waiting { what, prompt: false } => card(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(format!("Restarting when {what} finishes…"));
                if probed("guard-dont-restart", ui.button("Don't restart")).clicked() {
                    st.update.idle();
                }
            });
            theme::note(ui, "Closing this window calls the restart off.");
        }),
    }
}

/// A small coloured word beside a version.
fn badge(ui: &mut Ui, word: &str, colour: egui::Color32) {
    ui.label(RichText::new(word).size(11.0).color(colour));
}

/// One release (module doc).
fn entry(ui: &mut Ui, st: &mut UiState, r: &Release, acts: &mut Vec<Action>) {
    let mode = st.data_mode;
    let installed = r.version == st.update.current;
    let state = st.update.state_of(&r.tag);
    egui::Frame::new().fill(theme::PANEL_2).corner_radius(6).inner_margin(Margin::symmetric(10, 6)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            ui.label(RichText::new(&r.tag).size(14.5).strong());
            if !r.date.is_empty() {
                ui.label(RichText::new(&r.date).color(theme::INK_DIM));
            }
            if r.prerelease {
                badge(ui, "beta", AMBER);
            }
            if installed {
                badge(ui, "installed", GREEN);
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| match mode {
                DataMode::Dev => {}
                DataMode::Portable => {
                    if let Some(url) = zip_link(r) {
                        probed(&format!("zip-{}", r.tag), ui.hyperlink_to("Download zip", url)).on_hover_text("this release's portable zip, from GitHub, in your browser");
                    } else if let Some(page) = update::net::release_page(&r.tag) {
                        probed(&format!("page-{}", r.tag), ui.hyperlink_to("release page", page));
                    }
                }
                DataMode::Installed if installed => {}
                DataMode::Installed => match &state {
                    Some(DownloadState::Running(..)) => {
                        if probed("cancel-download", ui.button("Cancel")).on_hover_text("stop the download (nothing is installed)").clicked() {
                            acts.push(Action::CancelUpdateDownload);
                        }
                    }
                    Some(DownloadState::Ready(_)) => {
                        let b = Button::new(RichText::new("Restart now").strong().color(egui::Color32::WHITE)).fill(theme::GLOW);
                        if probed("restart-now", ui.add(b)).on_hover_text(format!("save the session, close, install {} and reopen paused where you left it (another open Atomic Sandbox window must be closed first)", r.tag)).clicked() {
                            acts.push(Action::RestartToUpdate);
                        }
                    }
                    _ => install_button(ui, st, r, acts),
                },
            });
        });
        if mode == DataMode::Installed && !installed {
            if let Err(why) = update::installable(r) {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&why).size(12.0).color(theme::INK_DIM));
                    if let Some(page) = update::net::release_page(&r.tag) {
                        ui.hyperlink_to(RichText::new("release page").size(12.0), page);
                    }
                });
            }
            match &state {
                Some(DownloadState::Running(got, total)) => {
                    let frac = if *total > 0 { *got as f32 / *total as f32 } else { 0.0 };
                    ui.add(egui::ProgressBar::new(frac).text(format!("downloading: {:.1} of {:.1} MB", *got as f64 / 1e6, *total as f64 / 1e6)));
                }
                Some(DownloadState::Ready(_)) => {
                    theme::note(ui, format!("{} is downloaded and verified (its SHA-256 and signature checked).", r.tag));
                }
                Some(DownloadState::Failed(e)) if e != update::apply::CANCELLED => {
                    ui.label(RichText::new(format!("the download failed: {e} — nothing was installed")).size(12.5).color(AMBER));
                }
                _ => {}
            }
            downgrade_card(ui, st, r, acts);
        }
        if !r.notes.trim().is_empty() {
            egui::CollapsingHeader::new(RichText::new("release notes").size(12.5)).id_salt(("release-notes", &r.tag)).default_open(false).show(ui, |ui| {
                ui.label(RichText::new(&r.notes).size(12.5));
            });
        }
    });
}

/// Install and restart (installed build): disabled with `installable`'s reason; an older target
/// asks first (its downgrade warnings).
fn install_button(ui: &mut Ui, st: &mut UiState, r: &Release, acts: &mut Vec<Action>) {
    let ok = update::installable(r);
    let older = r.version < st.update.current;
    let other = st.update.download.as_ref().filter(|d| d.tag() != r.tag && matches!(d.poll(), DownloadState::Running(..))).map(|d| d.tag().to_string());
    let mut tip = format!("download {}, verify it, then Restart now installs it", r.tag);
    if older {
        tip += " — an older version: its warnings about your data come first";
    }
    if let Some(t) = other {
        tip += &format!(" (this cancels the download of {t})");
    }
    let resp = probed(&format!("install-{}", r.tag), ui.add_enabled(ok.is_ok(), Button::new("Install and restart")));
    let resp = resp.on_hover_text(tip).on_disabled_hover_text(ok.err().unwrap_or_default());
    if resp.clicked() {
        if older {
            ask_downgrade(st, r);
        } else {
            acts.push(Action::InstallVersion(r.tag.clone()));
        }
    }
}

/// An older target's warnings, under its entry, before anything starts.
fn downgrade_card(ui: &mut Ui, st: &mut UiState, r: &Release, acts: &mut Vec<Action>) {
    let Some(c) = st.update.confirm.as_ref().filter(|c| c.tag == r.tag) else { return };
    let warnings = c.warnings.clone();
    let mut close = false;
    card(ui, |ui| {
        ui.label(RichText::new(format!("Before going back to the older {}:", r.tag)).strong());
        match &warnings {
            None => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(format!("reading {}'s data-format manifest…", r.tag));
                });
            }
            Some(w) if w.is_empty() => {
                theme::note(ui, format!("Nothing of yours is in a newer format than {} reads.", r.tag));
            }
            Some(w) => {
                for line in w {
                    ui.label(RichText::new(format!("• {line}")).size(12.5));
                }
                theme::note(ui, "Nothing is deleted, moved or converted: the older version just won't open what it can't read.");
            }
        }
        ui.horizontal(|ui| {
            if probed("install-anyway", ui.add_enabled(warnings.is_some(), Button::new(format!("Install {} anyway", r.tag)))).clicked() {
                acts.push(Action::InstallVersion(r.tag.clone()));
                close = true;
            }
            if probed("install-cancel", ui.button("Cancel")).clicked() {
                close = true;
            }
        });
    });
    if close {
        st.update.confirm = None;
    }
}
