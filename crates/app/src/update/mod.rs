//! The in-app updater's release logic (installer spec
//! `docs/superpowers/specs/2026-11-12-installer-updater-design.md`, plan Task 2): pure functions,
//! no network. The fetch and the download (Task 3) and the Version panel (Task 4) build on them.
//!
//! - [`release`]: GitHub's releases answer read into [`Release`]s, newest first; the Stable and
//!   Beta [`Channel`]s; the newer release on a channel; whether a release can be installed from
//!   the app (it must carry the installer, `SHA256SUMS.txt` and its minisign signature).
//! - [`manifest`]: a release's `manifest.json` — its version, channel, assets and the data
//!   formats it reads and writes.
//! - [`formats`]: the data-format numbers of this build, read from the code's own version
//!   constants, and the downgrade warnings drawn from them and the files under the per-user root.
//! - [`verify`]: an asset's SHA-256 checked against `SHA256SUMS.txt`, whose minisign signature is
//!   checked against the compiled-in release key first. Fails closed; a build without a real key
//!   (`crates/app/release-key.pub` holds a placeholder until Task 6) installs nothing.
//!
//! - [`net`] (Task 3): the HTTPS GETs — the releases list ([`fetch_releases`]) and the asset
//!   downloads — blocking, for worker threads only.
//! - [`apply`] (Task 3): the background [`Download`] (to `.part`, verified, then renamed), the
//!   [`Session`] saved before a restart and read back by `--resume-session`, and the hand-off to
//!   the silent installer ([`hand_off`]).
//!
//! Tests run offline against the recorded answers in `fixtures/` (that folder is `-text` in
//! `.gitattributes`: the signed bytes must not change at checkout); `net`'s against a loopback
//! server on 127.0.0.1.

pub mod apply;
pub mod formats;
pub mod manifest;
pub mod net;
pub mod release;
pub mod verify;

pub use apply::{hand_off, installer_command, load_session, save_session, session_path, CloseForUpdate, Download, DownloadState, Session};
pub use formats::{current_formats, downgrade_warnings, Formats};
pub use net::fetch_releases;
pub use manifest::{Manifest, ManifestAsset};
pub use release::{check_response, installable, newer, parse_releases, visible, Asset, Channel, CheckError, Release};
pub use verify::verify;

use std::path::Path;

/// `--write-manifest <path>` (hidden; CI, installer Task 6): this build's manifest
/// ([`Manifest::this_build`]: version, channel, data formats, no asset entries) written as JSON.
pub fn write_manifest(path: &Path) -> std::io::Result<()> {
    std::fs::write(path, Manifest::this_build().to_json())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `--write-manifest` writes valid JSON carrying this build's version and `current_formats()`,
    /// with no asset entries (CI adds them).
    #[test]
    fn write_manifest_writes_this_builds_formats() {
        let dir = std::env::temp_dir().join(format!("atomic-update-manifest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("manifest.json");
        write_manifest(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let json: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        assert_eq!(json["version"], env!("CARGO_PKG_VERSION"));
        let m = Manifest::parse(&text).unwrap();
        assert_eq!(m.formats, current_formats());
        assert!(m.assets.is_empty());
        assert_eq!(m.channel, if semver::Version::parse(env!("CARGO_PKG_VERSION")).unwrap().pre.is_empty() { "stable" } else { "beta" });
    }
}
