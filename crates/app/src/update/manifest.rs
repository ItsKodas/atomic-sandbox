//! A release's `manifest.json` (installer spec, "The manifest"): the version and channel, each
//! asset's name, size and SHA-256, and the data formats the build reads and writes
//! ([`super::formats`]). CI writes it with `app --write-manifest <path>` (no asset entries) and adds
//! the assets; the Version panel reads a target release's manifest before a downgrade. A release
//! without one (`v0.1.0`) records no formats ([`super::downgrade_warnings`] with `None`).
//!
//! Unknown fields are ignored and `assets` may be absent, so a later build's manifest still reads.

use super::formats::{current_formats, Formats};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// The version without the `v`, e.g. `0.2.0-beta.1`.
    pub version: String,
    /// `stable` or `beta` (a version with a pre-release part is `beta`).
    pub channel: String,
    #[serde(default)]
    pub assets: Vec<ManifestAsset>,
    /// Format name → version ([`super::formats`]' keys).
    pub formats: Formats,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ManifestAsset {
    pub name: String,
    /// Bytes.
    pub size: u64,
    /// Lower-case hex.
    pub sha256: String,
}

impl Manifest {
    /// This build's manifest: `CARGO_PKG_VERSION`, its channel, [`current_formats`], no assets.
    pub fn this_build() -> Manifest {
        let version = env!("CARGO_PKG_VERSION");
        let beta = semver::Version::parse(version).map(|v| !v.pre.is_empty()).unwrap_or(false);
        Manifest {
            version: version.to_string(),
            channel: if beta { "beta" } else { "stable" }.to_string(),
            assets: Vec::new(),
            formats: current_formats(),
        }
    }

    /// Reads a `manifest.json`; `Err` says what is wrong.
    pub fn parse(json: &str) -> Result<Manifest, String> {
        serde_json::from_str(json).map_err(|e| format!("manifest.json: {e}"))
    }

    /// Pretty-printed JSON with a trailing newline.
    pub fn to_json(&self) -> String {
        let mut s = serde_json::to_string_pretty(self).expect("a manifest always serialises");
        s.push('\n');
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const V020: &str = include_str!("fixtures/manifest_v0.2.0.json");

    #[test]
    fn manifest_round_trips() {
        let m = Manifest::parse(V020).unwrap();
        assert_eq!((m.version.as_str(), m.channel.as_str()), ("0.2.0", "stable"));
        assert_eq!(m.assets.len(), 2);
        assert_eq!(m.assets[0].name, "AtomicSandbox-Setup-0.2.0.exe");
        assert_eq!(m.assets[0].size, 41_234_567);
        assert_eq!(m.assets[0].sha256.len(), 64);
        assert_eq!(m.formats.get("collection"), Some(&1));
        assert_eq!(m.formats.get("sidecar"), Some(&8));
        assert_eq!(m.formats.len(), 8);
        assert_eq!(Manifest::parse(&m.to_json()).unwrap(), m, "written and read back unchanged");
        let built = Manifest::this_build();
        assert_eq!(Manifest::parse(&built.to_json()).unwrap(), built);
        // a later build's manifest with a field and a format this one doesn't know still reads
        let later = r#"{"version":"0.9.0","channel":"stable","signed_by":"x","formats":{"collection":2,"lattice":1}}"#;
        let l = Manifest::parse(later).unwrap();
        assert!(l.assets.is_empty());
        assert_eq!(l.formats.get("lattice"), Some(&1));
        assert!(Manifest::parse(r#"{"version":"0.2.0"}"#).is_err(), "formats are required");
    }
}
