//! The updater's network (installer spec, "The updater's network"; plan Task 3): HTTPS GET only,
//! of GitHub's releases list for this repository and of that repository's release assets — no
//! other host but GitHub's own asset CDN (spec amended 2026-11-13), nothing personal sent (the user
//! agent names the updater and its version only), no telemetry. Every call here blocks: it runs on
//! a worker or download thread, never the UI's.
//!
//! - [`fetch_releases`]: the releases list. An HTTP answer goes to [`check_response`] (status and
//!   body, so a rate limit or an HTML error page is a quiet message); a transport failure — no
//!   network, DNS, TLS, a timeout — is [`CheckError::Offline`].
//! - [`Http`]: the download job's byte [`Source`] (`apply`), refusing any URL that is not one of
//!   this repository's release assets.
//! - [`fetch_manifest`] (Task 4): a release's small `manifest.json`, in memory, through the same
//!   asset rules — the Version panel's downgrade warnings.
//!
//! **Redirects** are never followed by the client (`redirects(0)`) but by hand ([`follow`]): at
//! most [`MAX_REDIRECTS`] hops, each to an `https://` URL on one of [`ALLOWED_HOSTS`] exactly —
//! GitHub answers an asset URL with a redirect to its download CDN. Any other host is refused
//! ("download redirected to an unexpected host: <host>").
//!
//! `ureq` 2 over rustls with the bundled Mozilla roots (no native-tls, no system TLS settings):
//! 10 s to connect, 30 s for any one read — a slow download keeps going, a stalled one fails.

use super::apply::Source;
use super::manifest::Manifest;
use super::release::{check_response, Asset, CheckError, Release, MANIFEST_NAME};
use std::io::Read;
use std::time::Duration;

/// GitHub's releases list for this repository (unauthenticated: 60 requests an hour is plenty for
/// a check at launch and every 6 hours). `per_page=100`: every release in one answer.
// (the app's network, reached through `ui::version::GitHub`, which a test build never uses)
#[cfg_attr(test, allow(dead_code))]
pub const RELEASES_URL: &str = "https://api.github.com/repos/ItsKodas/atomic-sandbox/releases?per_page=100";

/// Every release asset of this repository is `<ASSET_PREFIX><tag>/<name>` ([`asset_url`]).
pub const ASSET_PREFIX: &str = "https://github.com/ItsKodas/atomic-sandbox/releases/download/";

/// The hosts a request or a redirect may reach, exactly (no subdomain, no port, no user info):
/// GitHub's API, github.com (the asset URLs), and GitHub's asset CDN, current and former.
pub const ALLOWED_HOSTS: [&str; 4] = ["api.github.com", "github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"];

/// The most redirects followed for one request.
pub const MAX_REDIRECTS: usize = 2;

pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
pub const READ_TIMEOUT: Duration = Duration::from_secs(30);

/// The user agent GitHub's API requires: the updater and this build's version, nothing else.
pub fn user_agent() -> String {
    format!("atomic-sandbox-updater/{}", env!("CARGO_PKG_VERSION"))
}

/// The updater's HTTP client: the timeouts, the user agent, HTTPS only, no redirect of its own
/// (they go through [`follow`]).
fn agent() -> ureq::Agent {
    builder().https_only(true).build()
}

fn builder() -> ureq::AgentBuilder {
    ureq::AgentBuilder::new().timeout_connect(CONNECT_TIMEOUT).timeout_read(READ_TIMEOUT).user_agent(&user_agent()).redirects(0)
}

/// What one request answered: the outcome, or a redirect to follow (its `Location`).
pub(crate) enum Hop<T> {
    Done(T),
    Redirect(String),
}

/// The lower-cased authority of an `https://` URL (its host, with any port or user info left in,
/// so that those fail the allowlist); `None` for any other scheme.
fn https_host(url: &str) -> Option<String> {
    let rest = url.get(..8).filter(|s| s.eq_ignore_ascii_case("https://")).map(|_| &url[8..])?;
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    Some(rest[..end].to_ascii_lowercase())
}

/// A `Location` against the URL that answered it: absolute as is, `//host/…` over HTTPS, `/path` on
/// the same origin; anything else is left for [`follow`] to refuse.
fn resolve(base: &str, location: &str) -> String {
    if location.starts_with("//") {
        format!("https:{location}")
    } else if location.starts_with('/') {
        match https_host(base) {
            Some(_) => {
                let authority_end = 8 + base[8..].find(['/', '?', '#']).unwrap_or(base.len() - 8);
                format!("{}{location}", &base[..authority_end])
            }
            None => location.to_string(),
        }
    } else {
        location.to_string()
    }
}

/// Runs `step` on `url`, following its redirects by hand (module doc): at most [`MAX_REDIRECTS`],
/// each to HTTPS on an [`ALLOWED_HOSTS`] host, else refused with why (`what` names the request).
/// The first URL is the caller's to vouch for.
pub(crate) fn follow<T>(what: &str, url: &str, mut step: impl FnMut(&str) -> Result<Hop<T>, String>) -> Result<T, String> {
    let mut url = url.to_string();
    let mut hops = 0;
    loop {
        match step(&url)? {
            Hop::Done(t) => return Ok(t),
            Hop::Redirect(location) => {
                if hops == MAX_REDIRECTS {
                    return Err(format!("{what} redirected more than {MAX_REDIRECTS} times"));
                }
                hops += 1;
                if location.is_empty() {
                    return Err(format!("{what} redirected without saying where"));
                }
                let next = resolve(&url, &location);
                match https_host(&next) {
                    None => return Err(format!("{what} redirected to a non-HTTPS address: {next}")),
                    Some(h) if !ALLOWED_HOSTS.contains(&h.as_str()) => return Err(format!("{what} redirected to an unexpected host: {h}")),
                    Some(_) => url = next,
                }
            }
        }
    }
}

/// A 3xx with a `Location`: the redirect; `None` otherwise.
fn redirect_of(r: &ureq::Response) -> Option<String> {
    (300..400).contains(&r.status()).then(|| r.header("location").unwrap_or("").to_string())
}

/// The releases list, newest first ([`check_response`]). Blocking: call it from a worker thread.
// (the app's network, reached through `ui::version::GitHub`, which a test build never uses)
#[cfg_attr(test, allow(dead_code))]
pub fn fetch_releases() -> Result<Vec<Release>, CheckError> {
    releases_from(&agent(), RELEASES_URL)
}

/// [`fetch_releases`] with its client and URL given (the tests use a loopback server).
fn releases_from(agent: &ureq::Agent, url: &str) -> Result<Vec<Release>, CheckError> {
    follow("GitHub's releases list", url, |u| Ok(releases_step(agent, u))).unwrap_or_else(|e| Err(CheckError::Malformed(e)))
}

fn releases_step(agent: &ureq::Agent, url: &str) -> Hop<Result<Vec<Release>, CheckError>> {
    match agent.get(url).set("Accept", "application/vnd.github+json").call() {
        Ok(r) => match redirect_of(&r) {
            Some(location) => Hop::Redirect(location),
            None => {
                let status = r.status();
                // a body cut off mid-way is the network's failure, not GitHub's answer
                Hop::Done(r.into_string().map_err(|_| CheckError::Offline).and_then(|body| check_response(status, &body)))
            }
        },
        // a refusal's body says why (a rate limit's JSON, or an HTML page): `check_response` reads it
        Err(ureq::Error::Status(status, r)) => Hop::Done(check_response(status, &r.into_string().unwrap_or_default())),
        Err(ureq::Error::Transport(_)) => Hop::Done(Err(CheckError::Offline)),
    }
}

/// `<ASSET_PREFIX><tag>/<name>`: the one URL a release's asset may have.
pub fn asset_url(tag: &str, name: &str) -> String {
    format!("{ASSET_PREFIX}{tag}/{name}")
}

/// A tag or asset name that stays one path segment as written, and one plain file name on disk:
/// not empty, no `/`, `\`, `:`, `%`, `?`, `#`, `..`, whitespace or control characters.
fn plain_segment(s: &str) -> bool {
    !s.is_empty() && !s.contains("..") && !s.chars().any(|c| matches!(c, '/' | '\\' | ':' | '%' | '?' | '#') || c.is_whitespace() || c.is_control())
}

/// `asset` is this release's own: its URL is exactly [`asset_url`] of the release's tag and the
/// asset's name, both plain segments.
pub fn check_asset(tag: &str, asset: &Asset) -> Result<(), String> {
    if !plain_segment(tag) || !plain_segment(&asset.name) {
        return Err(format!("the release {tag:?} or its asset {:?} has a name that isn't a plain file name", asset.name));
    }
    if asset.url != asset_url(tag, &asset.name) {
        return Err(format!("{}'s address {} isn't this release's ({})", asset.name, asset.url, asset_url(tag, &asset.name)));
    }
    Ok(())
}

/// Whether `url` is under this repository's release assets, as one `<tag>/<name>` (the only
/// downloads the updater starts).
pub fn is_release_asset(url: &str) -> bool {
    url.strip_prefix(ASSET_PREFIX).and_then(|rest| rest.split_once('/')).is_some_and(|(tag, name)| plain_segment(tag) && plain_segment(name))
}

/// The largest `manifest.json` taken (it is under a kilobyte).
const MANIFEST_MAX: u64 = 1 << 20;

/// A release's `manifest.json` (Task 4: the Version panel's downgrade warnings), small, in memory,
/// through the same rules as every download (the asset's URL tied to its release, the redirects
/// to GitHub's hosts only). `None` when the release carries none (`v0.1.0`). Blocking: call it from
/// a worker thread.
// (the app's network, reached through `ui::version::GitHub`, which a test build never uses)
#[cfg_attr(test, allow(dead_code))]
pub fn fetch_manifest(release: &Release) -> Result<Option<Manifest>, String> {
    manifest_from(release, &mut Http::new())
}

/// [`fetch_manifest`] from `source` (the tests serve it from memory).
pub(crate) fn manifest_from(release: &Release, source: &mut dyn Source) -> Result<Option<Manifest>, String> {
    let Some(a) = release.asset(MANIFEST_NAME) else { return Ok(None) };
    check_asset(&release.tag, a)?;
    let r = source.open(&a.url).map_err(|e| format!("couldn't download {}: {e}", a.name))?;
    let mut bytes = Vec::new();
    r.take(MANIFEST_MAX + 1).read_to_end(&mut bytes).map_err(|e| format!("the download of {} stopped part-way: {e}", a.name))?;
    if bytes.len() as u64 > MANIFEST_MAX {
        return Err(format!("{} is larger than {MANIFEST_MAX} bytes: not a release manifest", a.name));
    }
    let text = String::from_utf8(bytes).map_err(|_| format!("{} isn't text", a.name))?;
    Manifest::parse(&text).map(Some).map_err(|e| format!("{}: {e}", a.name))
}

/// The release's page on GitHub (`…/releases/tag/<tag>`), built from the tag — never from an
/// address in GitHub's answer — and only for a tag that is one plain path segment.
pub fn release_page(tag: &str) -> Option<String> {
    plain_segment(tag).then(|| format!("https://github.com/ItsKodas/atomic-sandbox/releases/tag/{tag}"))
}

/// The download job's source over HTTPS ([`super::apply::Download::start_with`], from
/// `ui::version::GitHub`).
pub struct Http {
    agent: ureq::Agent,
}

impl Http {
    pub fn new() -> Http {
        Http { agent: agent() }
    }
}

/// One asset request: the body, or the redirect to follow.
fn asset_step(agent: &ureq::Agent, url: &str) -> Result<Hop<Box<dyn Read + Send>>, String> {
    match agent.get(url).set("Accept", "application/octet-stream").call() {
        Ok(r) => Ok(match redirect_of(&r) {
            Some(location) => Hop::Redirect(location),
            None => Hop::Done(Box::new(r.into_reader())),
        }),
        Err(ureq::Error::Status(status, _)) => Err(format!("GitHub answered HTTP {status}")),
        Err(ureq::Error::Transport(e)) => Err(format!("couldn't download (offline?): {e}")),
    }
}

impl Source for Http {
    fn open(&mut self, url: &str) -> Result<Box<dyn Read + Send>, String> {
        if !is_release_asset(url) {
            return Err(format!("refused to download {url}: it isn't one of this repository's release assets"));
        }
        follow("download", url, |u| asset_step(&self.agent, u))
    }
}

#[cfg(test)]
mod tests {
    //! No request leaves the machine: the answers come from one-shot servers on 127.0.0.1, or from
    //! an injected redirect step.
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;

    const RELEASES: &str = include_str!("fixtures/releases.json");
    const RATE_LIMITED: &str = include_str!("fixtures/rate_limited.json");

    /// A loopback server answering one request with `status`, `headers` (each ending `\r\n`) and
    /// `body`; the receiver yields the request's head (the request line and headers, lower-cased).
    fn serve_once(status: &str, headers: &str, body: &'static str) -> (String, mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/repos/ItsKodas/atomic-sandbox/releases", listener.local_addr().unwrap());
        let (tx, rx) = mpsc::channel();
        let (status, headers) = (status.to_string(), headers.to_string());
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut head = String::new();
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap() == 0 || line == "\r\n" {
                    break;
                }
                head += &line.to_ascii_lowercase();
            }
            let mut stream = stream;
            write!(stream, "HTTP/1.1 {status}\r\n{headers}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
            let _ = tx.send(head);
        });
        (url, rx)
    }

    fn loopback_agent() -> ureq::Agent {
        builder().build() // plain HTTP to 127.0.0.1; the real agent is HTTPS only
    }

    /// A port with nothing listening on it.
    fn closed_port() -> u16 {
        TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port() // closed when dropped
    }

    /// A 200 is parsed as the releases list; the request names the updater and its version and
    /// asks for GitHub's JSON.
    #[test]
    fn a_releases_answer_is_parsed_and_the_request_is_plain() {
        let (url, head) = serve_once("200 OK", "", RELEASES);
        let rs = releases_from(&loopback_agent(), &url).unwrap();
        assert_eq!(rs.iter().map(|r| r.tag.as_str()).collect::<Vec<_>>(), ["v0.2.0", "v0.2.0-beta.1", "v0.1.0"]);
        let head = head.recv().unwrap();
        assert!(head.starts_with("get /repos/itskodas/atomic-sandbox/releases http/1.1\r\n"), "{head}");
        assert!(head.contains(&format!("user-agent: atomic-sandbox-updater/{}\r\n", env!("CARGO_PKG_VERSION"))), "{head}");
        assert!(head.contains("accept: application/vnd.github+json\r\n"), "{head}");
        assert!(!head.contains("authorization"), "unauthenticated: {head}");
    }

    /// A refusal's status and body reach `check_response`: 403 with GitHub's rate-limit JSON is
    /// the quiet rate-limit line, a 500 names its status.
    #[test]
    fn a_refusal_is_handed_to_check_response() {
        let (url, _head) = serve_once("403 Forbidden", "", RATE_LIMITED);
        assert_eq!(releases_from(&loopback_agent(), &url), Err(CheckError::RateLimited));
        let (url, _head) = serve_once("500 Internal Server Error", "", "<html>oops</html>");
        assert_eq!(releases_from(&loopback_agent(), &url), Err(CheckError::Http(500)));
    }

    /// Nothing listening (a closed port on 127.0.0.1) is offline, quietly.
    #[test]
    fn no_answer_is_offline() {
        let url = format!("http://127.0.0.1:{}/releases", closed_port());
        assert_eq!(releases_from(&loopback_agent(), &url), Err(CheckError::Offline));
        assert_eq!(CheckError::Offline.to_string(), "couldn't check (offline)");
    }

    /// The client follows no redirect itself: a 302 comes back as the redirect to judge (had it
    /// followed it, the closed port would have been a transport failure). Both requests' redirects
    /// to a host off the list are refused.
    #[test]
    fn the_client_follows_no_redirect_by_itself() {
        let target = format!("http://127.0.0.1:{}/elsewhere", closed_port());
        let (url, _head) = serve_once("302 Found", &format!("Location: {target}\r\n"), "");
        match asset_step(&loopback_agent(), &url) {
            Ok(Hop::Redirect(l)) => assert_eq!(l, target),
            Ok(Hop::Done(_)) => panic!("the redirect was followed"),
            Err(e) => panic!("{e}"),
        }
        let (url, _head) = serve_once("302 Found", "Location: https://evil.example/x\r\n", "");
        assert_eq!(follow("download", &url, |u| asset_step(&loopback_agent(), u)).err().unwrap(), "download redirected to an unexpected host: evil.example");
        let (url, _head) = serve_once("301 Moved Permanently", "Location: https://evil.example/r\r\n", "");
        assert_eq!(releases_from(&loopback_agent(), &url), Err(CheckError::Malformed("GitHub's releases list redirected to an unexpected host: evil.example".into())));
    }

    /// The injected redirect step: `chain` maps a URL to its redirect (absent: the final answer,
    /// its URL); the URLs asked for go to `seen`.
    fn walk(start: &str, chain: &[(&str, &str)]) -> (Result<String, String>, Vec<String>) {
        let mut seen = Vec::new();
        let r = follow("download", start, |u| {
            seen.push(u.to_string());
            Ok(match chain.iter().find(|(from, _)| *from == u) {
                Some((_, to)) => Hop::Redirect(to.to_string()),
                None => Hop::Done(u.to_string()),
            })
        });
        (r, seen)
    }

    /// GitHub's asset redirects are followed, by hand: to its CDN hosts over HTTPS, two hops at most;
    /// any other host, scheme or a third hop is refused before it is requested.
    #[test]
    fn redirects_are_followed_only_to_githubs_hosts() {
        let asset = "https://github.com/ItsKodas/atomic-sandbox/releases/download/v0.2.0/AtomicSandbox-Setup-0.2.0.exe";
        let cdn = "https://release-assets.githubusercontent.com/github-production-release-asset/1?sig=x";
        let (r, seen) = walk(asset, &[(asset, cdn)]);
        assert_eq!((r.unwrap(), seen.len()), (cdn.to_string(), 2));
        let old = "https://objects.githubusercontent.com/github-production-release-asset-2e65be/1";
        let (r, seen) = walk(asset, &[(asset, "https://GitHub.com/ItsKodas/atomic-sandbox/releases/download/v0.2.0/x"), ("https://GitHub.com/ItsKodas/atomic-sandbox/releases/download/v0.2.0/x", old)]);
        assert_eq!((r.unwrap(), seen.len()), (old.to_string(), 3), "two hops, host case-insensitive");
        // a relative Location stays on the same origin
        let (r, _) = walk(asset, &[(asset, "/ItsKodas/atomic-sandbox/releases/download/v0.2.0/y")]);
        assert_eq!(r.unwrap(), "https://github.com/ItsKodas/atomic-sandbox/releases/download/v0.2.0/y");
        let (r, _) = walk(asset, &[(asset, "//objects.githubusercontent.com/z")]);
        assert_eq!(r.unwrap(), "https://objects.githubusercontent.com/z");
        for (to, why) in [
            ("https://evil.example/a.exe", "download redirected to an unexpected host: evil.example"),
            ("https://github.com.evil.example/a", "download redirected to an unexpected host: github.com.evil.example"),
            ("https://x.release-assets.githubusercontent.com/a", "download redirected to an unexpected host: x.release-assets.githubusercontent.com"),
            ("https://github.com:8443/a", "download redirected to an unexpected host: github.com:8443"),
            ("https://evil.example@github.com/a", "download redirected to an unexpected host: evil.example@github.com"),
            ("http://release-assets.githubusercontent.com/a", "download redirected to a non-HTTPS address: http://release-assets.githubusercontent.com/a"),
            ("ftp://github.com/a", "download redirected to a non-HTTPS address: ftp://github.com/a"),
            ("", "download redirected without saying where"),
        ] {
            let (r, seen) = walk(asset, &[(asset, to)]);
            assert_eq!(r.err().as_deref(), Some(why), "{to}");
            assert_eq!(seen, [asset], "{to} was never requested");
        }
        let (a, b, c) = ("https://github.com/a", "https://github.com/b", "https://github.com/c");
        let (r, seen) = walk(asset, &[(asset, a), (a, b), (b, c)]);
        assert_eq!(r.err().as_deref(), Some("download redirected more than 2 times"));
        assert_eq!(seen, [asset, a, b], "the third hop is not requested");
    }

    /// Downloads go only to this repository's release assets, and each asset's URL must be exactly
    /// its own release's `<tag>/<name>`.
    #[test]
    fn only_this_repositorys_assets_are_downloaded() {
        assert!(is_release_asset("https://github.com/ItsKodas/atomic-sandbox/releases/download/v0.2.0/AtomicSandbox-Setup-0.2.0.exe"));
        for bad in [
            "http://github.com/ItsKodas/atomic-sandbox/releases/download/v0.2.0/a.exe",
            "https://github.com/someone/else/releases/download/v0.2.0/a.exe",
            "https://github.com/ItsKodas/atomic-sandbox/releases/download/",
            "https://github.com/ItsKodas/atomic-sandbox/releases/download/v1",
            "https://github.com/ItsKodas/atomic-sandbox/releases/download/../../x",
            "https://github.com/ItsKodas/atomic-sandbox/releases/download/v1/%2e%2e/a",
            "https://github.com/ItsKodas/atomic-sandbox/releases/download/v1/a/b.exe",
            "https://github.com/ItsKodas/atomic-sandbox/releases/download/v1/a.exe?x=1",
            "https://github.com.evil.example/ItsKodas/atomic-sandbox/releases/download/v1/a.exe",
        ] {
            assert!(!is_release_asset(bad), "{bad}");
        }
        let mut http = Http::new();
        assert!(http.open("https://example.com/a.exe").err().unwrap().contains("isn't one of this repository's release assets"));
        let asset = |name: &str, url: &str| Asset { name: name.into(), url: url.into(), size: 1 };
        let ok = "https://github.com/ItsKodas/atomic-sandbox/releases/download/v0.2.0/SHA256SUMS.txt";
        assert_eq!(check_asset("v0.2.0", &asset("SHA256SUMS.txt", ok)), Ok(()));
        assert!(check_asset("v0.1.0", &asset("SHA256SUMS.txt", ok)).unwrap_err().contains("isn't this release's"), "another release's sums");
        assert!(check_asset("v0.2.0", &asset("SHA256SUMS.txt", &ok.replace("SHA256SUMS.txt", "other.txt"))).is_err());
        for (tag, name) in [("v0.2.0", "a%2e.exe"), ("v0.2.0", ".."), ("v0.2.0", "a..exe"), ("v%30.2.0", "a.exe"), ("..", "a.exe"), ("v0.2.0", "a b.exe")] {
            let url = asset_url(tag, name);
            assert!(check_asset(tag, &asset(name, &url)).unwrap_err().contains("isn't a plain file name"), "{tag} {name}");
        }
    }

    /// Task 4: a release's `manifest.json` is read whole, in memory, from its own release's URL
    /// only; a release without one asks for nothing; a manifest that isn't one is refused; the
    /// release page is built from a plain tag only.
    #[test]
    fn a_release_manifest_is_read_through_the_asset_rules() {
        use crate::update::release::parse_releases;
        const MANIFEST: &str = include_str!("fixtures/manifest_v0.2.0.json");
        let releases = parse_releases(include_str!("fixtures/releases.json")).unwrap();
        let by = |t: &str| releases.iter().find(|r| r.tag == t).unwrap().clone();
        let asked = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let serve = |body: &'static str| {
            let log = asked.clone();
            move |url: &str| -> Result<Box<dyn Read + Send>, String> {
                log.lock().unwrap().push(url.to_string());
                Ok(Box::new(body.as_bytes()))
            }
        };
        let m = manifest_from(&by("v0.2.0"), &mut serve(MANIFEST)).unwrap().expect("v0.2.0 carries one");
        assert_eq!((m.version.as_str(), m.formats.get("sidecar")), ("0.2.0", Some(&8)));
        assert_eq!(asked.lock().unwrap().as_slice(), ["https://github.com/ItsKodas/atomic-sandbox/releases/download/v0.2.0/manifest.json"]);
        asked.lock().unwrap().clear();
        assert_eq!(manifest_from(&by("v0.1.0"), &mut serve(MANIFEST)), Ok(None));
        assert!(asked.lock().unwrap().is_empty(), "no manifest asset: nothing asked");
        assert!(manifest_from(&by("v0.2.0"), &mut serve("<html>not it</html>")).unwrap_err().starts_with("manifest.json: "));
        // another address than its release's own: refused before any request
        let mut moved = by("v0.2.0");
        moved.assets.iter_mut().find(|a| a.name == MANIFEST_NAME).unwrap().url = "https://example.com/manifest.json".into();
        asked.lock().unwrap().clear();
        assert!(manifest_from(&moved, &mut serve(MANIFEST)).unwrap_err().contains("isn't this release's"));
        assert!(asked.lock().unwrap().is_empty());
        assert_eq!(release_page("v0.2.0").as_deref(), Some("https://github.com/ItsKodas/atomic-sandbox/releases/tag/v0.2.0"));
        assert_eq!(release_page("../x"), None);
    }
}
