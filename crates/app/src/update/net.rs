//! The updater's network (installer spec, "The updater's network"; plan Task 3): HTTPS GET only,
//! of GitHub's releases list for this repository and of that repository's release assets — no
//! other host, nothing personal sent (the user agent names the updater and its version only), no
//! telemetry. Every call here blocks: it runs on a worker or download thread, never the UI's.
//!
//! - [`fetch_releases`]: the releases list. An HTTP answer goes to [`check_response`] (status and
//!   body, so a rate limit or an HTML error page is a quiet message); a transport failure — no
//!   network, DNS, TLS, a timeout — is [`CheckError::Offline`].
//! - [`Http`]: the download job's byte [`Source`] (`apply`), refusing any URL that is not one of
//!   this repository's release assets. GitHub answers those with a redirect to its own download
//!   host, which is followed (HTTPS only).
//!
//! `ureq` 2 over rustls with the bundled Mozilla roots (no native-tls, no system TLS settings):
//! 10 s to connect, 30 s for any one read — a slow download keeps going, a stalled one fails.

use super::apply::Source;
use super::release::{check_response, CheckError, Release};
use std::io::Read;
use std::time::Duration;

/// GitHub's releases list for this repository (unauthenticated: 60 requests an hour is plenty for
/// a check at launch and every 6 hours). `per_page=100`: every release in one answer.
pub const RELEASES_URL: &str = "https://api.github.com/repos/ItsKodas/atomic-sandbox/releases?per_page=100";

/// Every release asset of this repository is under this prefix (`browser_download_url`).
pub const ASSET_PREFIX: &str = "https://github.com/ItsKodas/atomic-sandbox/releases/download/";

pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
pub const READ_TIMEOUT: Duration = Duration::from_secs(30);

/// The user agent GitHub's API requires: the updater and this build's version, nothing else.
pub fn user_agent() -> String {
    format!("atomic-sandbox-updater/{}", env!("CARGO_PKG_VERSION"))
}

/// The updater's HTTP client: the timeouts, the user agent, HTTPS only (a redirect to plain HTTP
/// is refused).
fn agent() -> ureq::Agent {
    builder().https_only(true).build()
}

fn builder() -> ureq::AgentBuilder {
    ureq::AgentBuilder::new().timeout_connect(CONNECT_TIMEOUT).timeout_read(READ_TIMEOUT).user_agent(&user_agent())
}

/// The releases list, newest first ([`check_response`]). Blocking: call it from a worker thread.
pub fn fetch_releases() -> Result<Vec<Release>, CheckError> {
    releases_from(&agent(), RELEASES_URL)
}

/// [`fetch_releases`] with its client and URL given (the tests use a loopback server).
fn releases_from(agent: &ureq::Agent, url: &str) -> Result<Vec<Release>, CheckError> {
    match agent.get(url).set("Accept", "application/vnd.github+json").call() {
        Ok(r) => {
            let status = r.status();
            // a body cut off mid-way is the network's failure, not GitHub's answer
            let body = r.into_string().map_err(|_| CheckError::Offline)?;
            check_response(status, &body)
        }
        // a refusal's body says why (a rate limit's JSON, or an HTML page): `check_response` reads it
        Err(ureq::Error::Status(status, r)) => check_response(status, &r.into_string().unwrap_or_default()),
        Err(ureq::Error::Transport(_)) => Err(CheckError::Offline),
    }
}

/// Whether `url` is one of this repository's release assets (the only downloads the updater makes).
pub fn is_release_asset(url: &str) -> bool {
    url.strip_prefix(ASSET_PREFIX).is_some_and(|rest| !rest.is_empty() && !rest.contains("..") && !rest.contains(['?', '#', '\\']))
}

/// The download job's source over HTTPS ([`super::apply::Download::start`]).
pub struct Http {
    agent: ureq::Agent,
}

impl Http {
    pub fn new() -> Http {
        Http { agent: agent() }
    }
}

impl Source for Http {
    fn open(&mut self, url: &str) -> Result<Box<dyn Read + Send>, String> {
        if !is_release_asset(url) {
            return Err(format!("refused to download {url}: it isn't one of this repository's release assets"));
        }
        match self.agent.get(url).set("Accept", "application/octet-stream").call() {
            Ok(r) => Ok(Box::new(r.into_reader())),
            Err(ureq::Error::Status(status, _)) => Err(format!("GitHub answered HTTP {status}")),
            Err(ureq::Error::Transport(e)) => Err(format!("couldn't download (offline?): {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    //! No request leaves the machine: the answers come from a one-shot server on 127.0.0.1.
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;

    const RELEASES: &str = include_str!("fixtures/releases.json");
    const RATE_LIMITED: &str = include_str!("fixtures/rate_limited.json");

    /// A loopback server answering one request with `status` and `body`; the receiver yields the
    /// request's head (the request line and headers, lower-cased).
    fn serve_once(status: &str, body: &'static str) -> (String, mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/repos/ItsKodas/atomic-sandbox/releases", listener.local_addr().unwrap());
        let (tx, rx) = mpsc::channel();
        let status = status.to_string();
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
            write!(stream, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
            let _ = tx.send(head);
        });
        (url, rx)
    }

    fn loopback_agent() -> ureq::Agent {
        builder().build() // plain HTTP to 127.0.0.1; the real agent is HTTPS only
    }

    /// A 200 is parsed as the releases list; the request names the updater and its version and
    /// asks for GitHub's JSON.
    #[test]
    fn a_releases_answer_is_parsed_and_the_request_is_plain() {
        let (url, head) = serve_once("200 OK", RELEASES);
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
        let (url, _head) = serve_once("403 Forbidden", RATE_LIMITED);
        assert_eq!(releases_from(&loopback_agent(), &url), Err(CheckError::RateLimited));
        let (url, _head) = serve_once("500 Internal Server Error", "<html>oops</html>");
        assert_eq!(releases_from(&loopback_agent(), &url), Err(CheckError::Http(500)));
    }

    /// Nothing listening (a closed port on 127.0.0.1) is offline, quietly.
    #[test]
    fn no_answer_is_offline() {
        let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port(); // closed when dropped
        let url = format!("http://127.0.0.1:{port}/releases");
        assert_eq!(releases_from(&loopback_agent(), &url), Err(CheckError::Offline));
        assert_eq!(CheckError::Offline.to_string(), "couldn't check (offline)");
    }

    /// Downloads go only to this repository's release assets.
    #[test]
    fn only_this_repositorys_assets_are_downloaded() {
        assert!(is_release_asset("https://github.com/ItsKodas/atomic-sandbox/releases/download/v0.2.0/AtomicSandbox-Setup-0.2.0.exe"));
        for bad in [
            "http://github.com/ItsKodas/atomic-sandbox/releases/download/v0.2.0/a.exe",
            "https://github.com/someone/else/releases/download/v0.2.0/a.exe",
            "https://github.com/ItsKodas/atomic-sandbox/releases/download/",
            "https://github.com/ItsKodas/atomic-sandbox/releases/download/../../x",
            "https://github.com/ItsKodas/atomic-sandbox/releases/download/v1/a.exe?x=1",
            "https://github.com.evil.example/ItsKodas/atomic-sandbox/releases/download/v1/a.exe",
        ] {
            assert!(!is_release_asset(bad), "{bad}");
        }
        let mut http = Http::new();
        assert!(http.open("https://example.com/a.exe").err().unwrap().contains("isn't one of this repository's release assets"));
    }
}
