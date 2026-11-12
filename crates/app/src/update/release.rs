//! GitHub's releases answer (`GET /repos/ItsKodas/atomic-sandbox/releases`) read into
//! [`Release`]s, and the choices the Version panel makes from them (module doc of `update`).
//!
//! Tags follow semver with a leading `v`: `vX.Y.Z` is a release, `vX.Y.Z-beta.N` a pre-release
//! (installer spec, "Versions"). A tag that is not a version is skipped, not an error: one odd
//! release never hides the rest. Drafts are skipped (they are not public). A release counts as a
//! pre-release when GitHub marks it so **or** its version has a pre-release part, so a mis-marked
//! beta never reaches the Stable channel.

use serde::Deserialize;

/// The asset every installable release carries beside its installer: `<sha256>  <name>` lines.
pub const SUMS_NAME: &str = "SHA256SUMS.txt";
/// The minisign signature over [`SUMS_NAME`].
pub const SIG_NAME: &str = "SHA256SUMS.txt.minisig";
/// The release's data-format manifest (`update::manifest`); a release without one (`v0.1.0`)
/// records no formats.
pub const MANIFEST_NAME: &str = "manifest.json";
/// [`installable`]'s refusal of a release without `SHA256SUMS.txt` and its signature (review focus
/// 5: such a release is listed, never installed unverified).
pub const NO_SIGNATURE: &str = "can't be installed from the app (no signature) — download it from GitHub";
/// [`installable`]'s refusal of a signed release without its installer.
pub const NO_INSTALLER: &str = "can't be installed from the app (no installer) — download it from GitHub";

/// The installer's asset name for a version: `AtomicSandbox-Setup-<ver>.exe` (`<ver>` without the `v`).
pub fn installer_name(v: &semver::Version) -> String {
    format!("AtomicSandbox-Setup-{v}.exe")
}

/// Which releases the panel lists and offers (a display setting stored in the profile, Task 4).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Channel {
    /// Releases only.
    #[default]
    Stable,
    /// Pre-releases too.
    Beta,
}

/// One published GitHub release.
#[derive(Clone, Debug, PartialEq)]
pub struct Release {
    /// The tag as GitHub has it, e.g. `v0.2.0-beta.1`.
    pub tag: String,
    /// The tag's version (without the `v`).
    pub version: semver::Version,
    /// GitHub marks it a pre-release, or its version has a pre-release part (module doc).
    pub prerelease: bool,
    /// The publication date, `YYYY-MM-DD` (empty when GitHub gives none).
    pub date: String,
    /// The release notes (Markdown as written; empty when there are none).
    pub notes: String,
    pub assets: Vec<Asset>,
}

/// One file attached to a release.
#[derive(Clone, Debug, PartialEq)]
pub struct Asset {
    pub name: String,
    /// `browser_download_url`.
    pub url: String,
    /// Bytes.
    pub size: u64,
}

impl Release {
    /// The asset of this name, if the release carries it.
    pub fn asset(&self, name: &str) -> Option<&Asset> {
        self.assets.iter().find(|a| a.name == name)
    }
}

/// Why a check found no list. Each `Display` is the panel's quiet one-line message (no dialog,
/// ever; installer spec "The updater's network").
#[derive(Clone, Debug, PartialEq)]
pub enum CheckError {
    /// No connection (DNS, connect or TLS failure; Task 3 decides which errors are this).
    Offline,
    /// GitHub refused or limited the request (HTTP 403 or 429, or its rate-limit message).
    RateLimited,
    /// Any other non-success status.
    Http(u16),
    /// The answer is not a releases list (not JSON, or not the expected shape).
    Malformed(String),
}

impl std::fmt::Display for CheckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckError::Offline => write!(f, "couldn't check (offline)"),
            CheckError::RateLimited => write!(f, "couldn't check (GitHub said: rate limited, try later)"),
            CheckError::Http(code) => write!(f, "couldn't check (GitHub answered HTTP {code})"),
            CheckError::Malformed(why) => write!(f, "couldn't check (GitHub's answer wasn't a releases list: {why})"),
        }
    }
}

impl std::error::Error for CheckError {}

#[derive(Deserialize)]
struct GhRelease {
    tag_name: String,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
    #[serde(default)]
    published_at: Option<String>,
    #[serde(default)]
    created_at: Option<String>,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    assets: Vec<GhAsset>,
}

#[derive(Deserialize)]
struct GhAsset {
    name: String,
    browser_download_url: String,
    #[serde(default)]
    size: u64,
}

/// The tag's version: `v` (or `V`) then semver.
fn tag_version(tag: &str) -> Option<semver::Version> {
    let t = tag.strip_prefix('v').or_else(|| tag.strip_prefix('V'))?;
    semver::Version::parse(t).ok()
}

/// GitHub's releases answer (a 200 body) read into the public releases, newest version first
/// (module doc). An error object instead of a list (`{"message": …}`) is [`CheckError::RateLimited`]
/// when it says so, else [`CheckError::Malformed`] carrying GitHub's message.
pub fn parse_releases(json: &str) -> Result<Vec<Release>, CheckError> {
    let value: serde_json::Value = serde_json::from_str(json).map_err(|e| CheckError::Malformed(format!("not JSON ({e})")))?;
    if let Some(obj) = value.as_object() {
        let message = obj.get("message").and_then(|m| m.as_str()).unwrap_or("an object, not a list");
        if message.to_ascii_lowercase().contains("rate limit") {
            return Err(CheckError::RateLimited);
        }
        return Err(CheckError::Malformed(format!("GitHub said: {message}")));
    }
    let raw: Vec<GhRelease> = serde_json::from_value(value).map_err(|e| CheckError::Malformed(e.to_string()))?;
    let mut out: Vec<Release> = raw
        .into_iter()
        .filter(|r| !r.draft)
        .filter_map(|r| {
            let version = tag_version(&r.tag_name)?;
            let date = r.published_at.or(r.created_at).map(|d| d.chars().take(10).collect()).unwrap_or_default();
            Some(Release {
                prerelease: r.prerelease || !version.pre.is_empty(),
                tag: r.tag_name,
                version,
                date,
                notes: r.body.unwrap_or_default(),
                assets: r.assets.into_iter().map(|a| Asset { name: a.name, url: a.browser_download_url, size: a.size }).collect(),
            })
        })
        .collect();
    out.sort_by(|a, b| b.version.cmp(&a.version));
    Ok(out)
}

/// A whole HTTP answer to the releases request (Task 3's fetch hands it here): 403 and 429 are
/// [`CheckError::RateLimited`] whatever the body (review focus 1: GitHub's secondary limit may not
/// even be JSON), another non-2xx status is [`CheckError::Http`], a 2xx body is [`parse_releases`].
pub fn check_response(status: u16, body: &str) -> Result<Vec<Release>, CheckError> {
    match status {
        200..=299 => parse_releases(body),
        403 | 429 => Err(CheckError::RateLimited),
        s => Err(CheckError::Http(s)),
    }
}

/// The releases a channel lists, in `releases`' order: [`Channel::Stable`] hides pre-releases.
pub fn visible(releases: &[Release], ch: Channel) -> Vec<&Release> {
    releases.iter().filter(|r| ch == Channel::Beta || !r.prerelease).collect()
}

/// The newest release on the channel that is newer than `current` (semver precedence: a beta is
/// older than its release), if any: what the toolbar chip offers.
pub fn newer<'a>(releases: &'a [Release], ch: Channel, current: &semver::Version) -> Option<&'a Release> {
    visible(releases, ch).into_iter().filter(|r| r.version > *current).max_by(|a, b| a.version.cmp(&b.version))
}

/// The installer, `SHA256SUMS.txt` and `SHA256SUMS.txt.minisig` of a release the app can
/// install, or why it can't (the panel shows the reason with a link to the release page). A
/// release without the sums and their signature is never installable: it would install
/// unverified (installer spec, "Verification fails closed").
pub fn installable(r: &Release) -> Result<(&Asset, &Asset, &Asset), String> {
    let (Some(sums), Some(sig)) = (r.asset(SUMS_NAME), r.asset(SIG_NAME)) else {
        return Err(NO_SIGNATURE.into());
    };
    let setup = r.asset(&installer_name(&r.version)).ok_or_else(|| NO_INSTALLER.to_string())?;
    Ok((setup, sums, sig))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RELEASES: &str = include_str!("fixtures/releases.json");
    const RATE_LIMITED: &str = include_str!("fixtures/rate_limited.json");

    fn v(s: &str) -> semver::Version {
        semver::Version::parse(s).unwrap()
    }

    #[test]
    fn parses_the_recorded_releases_newest_first() {
        let rs = parse_releases(RELEASES).unwrap();
        let tags: Vec<&str> = rs.iter().map(|r| r.tag.as_str()).collect();
        assert_eq!(tags, ["v0.2.0", "v0.2.0-beta.1", "v0.1.0"], "the draft v0.3.0 is skipped");
        let r = &rs[0];
        assert_eq!((r.version.clone(), r.prerelease, r.date.as_str()), (v("0.2.0"), false, "2026-12-01"));
        assert!(r.notes.contains("Installer"), "{}", r.notes);
        assert_eq!(r.assets.len(), 5);
        let setup = r.asset("AtomicSandbox-Setup-0.2.0.exe").unwrap();
        assert_eq!(setup.size, 41_234_567);
        assert_eq!(setup.url, "https://github.com/ItsKodas/atomic-sandbox/releases/download/v0.2.0/AtomicSandbox-Setup-0.2.0.exe");
        assert!(rs[1].prerelease);
        assert_eq!((rs[2].version.clone(), rs[2].prerelease, rs[2].notes.as_str()), (v("0.1.0"), false, ""), "a null body is empty notes");
    }

    #[test]
    fn stable_hides_prereleases_beta_shows_them() {
        let rs = parse_releases(RELEASES).unwrap();
        let tags = |ch| visible(&rs, ch).iter().map(|r| r.tag.clone()).collect::<Vec<_>>();
        assert_eq!(tags(Channel::Stable), ["v0.2.0", "v0.1.0"]);
        assert_eq!(tags(Channel::Beta), ["v0.2.0", "v0.2.0-beta.1", "v0.1.0"]);
        // a beta GitHub forgot to mark is still a pre-release (its version says so)
        let json = r#"[{"tag_name":"v0.4.0-beta.2","prerelease":false,"draft":false,"assets":[]}]"#;
        let unmarked = parse_releases(json).unwrap();
        assert!(unmarked[0].prerelease);
        assert!(visible(&unmarked, Channel::Stable).is_empty());
    }

    #[test]
    fn newer_is_none_when_current_is_latest() {
        let rs = parse_releases(RELEASES).unwrap();
        assert_eq!(newer(&rs, Channel::Stable, &v("0.2.0")), None);
        assert_eq!(newer(&rs, Channel::Beta, &v("0.2.0")), None);
        assert_eq!(newer(&rs, Channel::Stable, &v("0.1.0")).map(|r| r.tag.as_str()), Some("v0.2.0"));
        assert_eq!(newer(&rs, Channel::Beta, &v("0.1.0")).map(|r| r.tag.as_str()), Some("v0.2.0"), "the release outranks its beta");
        assert_eq!(newer(&rs, Channel::Stable, &v("0.2.0-beta.1")).map(|r| r.tag.as_str()), Some("v0.2.0"));
        // only the beta is newer: Beta offers it, Stable doesn't
        let early: Vec<Release> = rs.iter().filter(|r| r.tag != "v0.2.0").cloned().collect();
        assert_eq!(newer(&early, Channel::Beta, &v("0.1.0")).map(|r| r.tag.as_str()), Some("v0.2.0-beta.1"));
        assert_eq!(newer(&early, Channel::Stable, &v("0.1.0")), None);
    }

    #[test]
    fn a_rate_limited_response_is_a_quiet_message() {
        let e = parse_releases(RATE_LIMITED).unwrap_err();
        assert_eq!(e, CheckError::RateLimited);
        assert!(e.to_string().contains("rate limited"), "{e}");
        // the whole answer: 403 and 429 are rate limits whatever the body, even HTML
        assert_eq!(check_response(403, RATE_LIMITED), Err(CheckError::RateLimited));
        assert_eq!(check_response(429, "<html>slow down</html>"), Err(CheckError::RateLimited));
        assert_eq!(check_response(502, ""), Err(CheckError::Http(502)));
        assert_eq!(check_response(200, RELEASES).map(|r| r.len()), Ok(3));
        // a non-JSON 200 or another error object is malformed, never a crash
        assert!(matches!(check_response(200, "<html>"), Err(CheckError::Malformed(_))));
        let other = parse_releases(r#"{"message":"Not Found"}"#).unwrap_err();
        assert!(other.to_string().contains("Not Found"), "{other}");
        for e in [CheckError::Offline, CheckError::RateLimited, CheckError::Http(500), CheckError::Malformed("x".into())] {
            assert!(e.to_string().starts_with("couldn't check ("), "{e}");
        }
        assert_eq!(CheckError::Offline.to_string(), "couldn't check (offline)");
    }

    #[test]
    fn an_unsigned_release_is_listed_but_not_installable() {
        let rs = parse_releases(RELEASES).unwrap();
        let old = rs.iter().find(|r| r.tag == "v0.1.0").expect("v0.1.0 is listed");
        assert!(visible(&rs, Channel::Stable).contains(&old));
        assert_eq!(installable(old), Err(NO_SIGNATURE.to_string()));
        assert_eq!(NO_SIGNATURE, "can't be installed from the app (no signature) — download it from GitHub");
        let (setup, sums, sig) = installable(&rs[0]).unwrap();
        assert_eq!((setup.name.as_str(), sums.name.as_str(), sig.name.as_str()), ("AtomicSandbox-Setup-0.2.0.exe", SUMS_NAME, SIG_NAME));
        let beta = installable(&rs[1]).unwrap();
        assert_eq!(beta.0.name, "AtomicSandbox-Setup-0.2.0-beta.1.exe");
        // signed, but without its installer: listed, not installable
        let mut no_setup = rs[0].clone();
        no_setup.assets.retain(|a| !a.name.ends_with(".exe"));
        assert_eq!(installable(&no_setup), Err(NO_INSTALLER.to_string()));
        // sums without their signature is unsigned
        let mut no_sig = rs[0].clone();
        no_sig.assets.retain(|a| a.name != SIG_NAME);
        assert_eq!(installable(&no_sig), Err(NO_SIGNATURE.to_string()));
    }
}
