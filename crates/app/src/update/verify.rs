//! An asset checked before anything is installed (installer spec, "Verification fails closed"):
//! `SHA256SUMS.txt` must carry a valid minisign signature under the compiled-in release key, and
//! must list the asset with the SHA-256 its bytes have. Any doubt is a refusal: a key that doesn't
//! decode, a signature that doesn't decode or match (legacy, non-prehashed signatures included), a
//! sums file that isn't text, an asset listed nowhere or twice with different sums, a mismatch.
//!
//! The release key (`crates/app/release-key.pub`, compiled in as [`RELEASE_KEY`]) is a placeholder
//! comment until installer Task 6; a build carrying it installs nothing ([`NO_RELEASE_KEY`]). The
//! tests sign with the committed `fixtures/TEST-ONLY.*` pair, which signs nothing real.

use sha2::{Digest, Sha256};

/// The public key releases are signed with: the text of `crates/app/release-key.pub` (minisign's
/// two-line `.pub` form, or `#` comment lines only while there is no key yet).
pub const RELEASE_KEY: &str = include_str!("../../release-key.pub");

/// [`verify`]'s refusal when the key holds no key (the placeholder).
pub const NO_RELEASE_KEY: &str = "this build has no release key";

/// SHA-256 of `bytes`.
pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Lower-case hex of a digest.
pub fn hex(d: &[u8; 32]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Option<[u8; 32]> {
    // hex digits only (`from_str_radix` alone would take a leading `+`)
    if s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

/// The key's base64 line: the first line that is neither blank, a `#` comment nor minisign's
/// `untrusted comment:` line. `None` = no key (the placeholder).
fn key_line(pubkey: &str) -> Option<&str> {
    pubkey.lines().map(str::trim).find(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("untrusted comment:"))
}

/// Whether `pubkey` holds a key at all (not the placeholder): the download is refused before
/// anything is fetched without one (`apply::Download`). [`verify`] still decides.
pub fn has_key(pubkey: &str) -> bool {
    key_line(pubkey).is_some()
}

/// `sums` is signed by `pubkey` (`sig`: the `.minisig` file's bytes) and lists `file_name` with
/// `file_bytes_sha256`: `Ok(())`, else why not (module doc). The signature is checked before the
/// sums are read.
pub fn verify(sums: &[u8], sig: &[u8], file_name: &str, file_bytes_sha256: &[u8; 32], pubkey: &str) -> Result<(), String> {
    let line = key_line(pubkey).ok_or_else(|| NO_RELEASE_KEY.to_string())?;
    let pk = minisign_verify::PublicKey::from_base64(line).map_err(|e| format!("this build's release key doesn't decode ({e})"))?;
    let sig_text = std::str::from_utf8(sig).map_err(|_| "SHA256SUMS.txt.minisig isn't a minisign signature (not text)".to_string())?;
    let signature = minisign_verify::Signature::decode(sig_text).map_err(|e| format!("SHA256SUMS.txt.minisig isn't a minisign signature ({e})"))?;
    pk.verify(sums, &signature, false).map_err(|e| format!("SHA256SUMS.txt isn't signed by this build's release key ({e})"))?;
    let text = std::str::from_utf8(sums).map_err(|_| "SHA256SUMS.txt isn't text".to_string())?;
    // `<64 hex>  <name>` (text mode) or `<64 hex> *<name>` (binary mode), as sha256sum writes them
    let mut listed: Option<[u8; 32]> = None;
    for l in text.lines() {
        let Some((sum, rest)) = l.split_once(' ') else { continue };
        let entry = rest.strip_prefix(' ').or_else(|| rest.strip_prefix('*')).unwrap_or(rest);
        if entry != file_name {
            continue;
        }
        let d = unhex(&sum.to_ascii_lowercase()).ok_or_else(|| format!("SHA256SUMS.txt's entry for {file_name} isn't a SHA-256"))?;
        if listed.is_some_and(|p| p != d) {
            return Err(format!("SHA256SUMS.txt lists {file_name} twice with different sums"));
        }
        listed = Some(d);
    }
    let want = listed.ok_or_else(|| format!("{file_name} isn't listed in SHA256SUMS.txt"))?;
    if &want != file_bytes_sha256 {
        return Err(format!("{file_name} doesn't match SHA256SUMS.txt (its SHA-256 is {}, the release says {})", hex(file_bytes_sha256), hex(&want)));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bytes the fixture `SHA256SUMS.txt` lists as the v0.2.0 installer (no real installer is
    /// committed; [`regenerate_signed_fixtures`] wrote the sums from these).
    pub(super) const FIXTURE_SETUP: &[u8] = b"AtomicSandbox-Setup-0.2.0.exe: a stand-in for the installer, test only\n";
    const FIXTURE_ZIP: &[u8] = b"AtomicSandbox-0.2.0-windows-x64.zip: a stand-in for the zip, test only\n";
    const SETUP: &str = "AtomicSandbox-Setup-0.2.0.exe";
    const SUMS: &[u8] = include_bytes!("fixtures/SHA256SUMS.txt");
    const SIG: &[u8] = include_bytes!("fixtures/SHA256SUMS.txt.minisig");
    const TEST_KEY: &str = include_str!("fixtures/TEST-ONLY.pub");

    #[test]
    fn verify_accepts_the_fixture_and_rejects_a_flipped_byte() {
        let good = sha256(FIXTURE_SETUP);
        assert_eq!(verify(SUMS, SIG, SETUP, &good, TEST_KEY), Ok(()));
        assert_eq!(verify(SUMS, SIG, "AtomicSandbox-0.2.0-windows-x64.zip", &sha256(FIXTURE_ZIP), TEST_KEY), Ok(()));
        // one flipped byte of the download: its SHA-256 no longer matches
        let mut flipped = FIXTURE_SETUP.to_vec();
        flipped[3] ^= 1;
        let e = verify(SUMS, SIG, SETUP, &sha256(&flipped), TEST_KEY).unwrap_err();
        assert!(e.contains("doesn't match SHA256SUMS.txt"), "{e}");
        // an asset the sums don't list
        let e = verify(SUMS, SIG, "AtomicSandbox-Setup-0.2.1.exe", &good, TEST_KEY).unwrap_err();
        assert!(e.contains("isn't listed"), "{e}");
        // one flipped byte of the sums themselves: the signature fails first
        let mut sums = SUMS.to_vec();
        sums[0] ^= 1;
        let e = verify(&sums, SIG, SETUP, &good, TEST_KEY).unwrap_err();
        assert!(e.contains("isn't signed"), "{e}");
    }

    #[test]
    fn verify_rejects_a_bad_signature() {
        let good = sha256(FIXTURE_SETUP);
        // the signature's own bytes corrupted (one base64 character of the signature line)
        let text = std::str::from_utf8(SIG).unwrap();
        let mut lines: Vec<String> = text.lines().map(String::from).collect();
        let sig_line = &mut lines[1];
        let mid = sig_line.len() / 2;
        let c = sig_line.as_bytes()[mid];
        sig_line.replace_range(mid..mid + 1, if c == b'A' { "B" } else { "A" });
        let bad = lines.join("\n") + "\n";
        assert!(verify(SUMS, bad.as_bytes(), SETUP, &good, TEST_KEY).is_err());
        // a changed trusted comment breaks the global signature
        let mut lines: Vec<String> = text.lines().map(String::from).collect();
        lines[2].push('!');
        let bad = lines.join("\n") + "\n";
        assert!(verify(SUMS, bad.as_bytes(), SETUP, &good, TEST_KEY).is_err());
        // garbage, an empty signature, another key
        assert!(verify(SUMS, b"not a signature", SETUP, &good, TEST_KEY).is_err());
        assert!(verify(SUMS, b"", SETUP, &good, TEST_KEY).is_err());
        assert!(verify(SUMS, &[0xff, 0xfe], SETUP, &good, TEST_KEY).is_err());
        let other_key = "untrusted comment: another key\nRWQf6LRCGA9i53mlYecO4IzT51TGPpvWucNSCh1CBM0QTaLn73Y7GFO3\n";
        let e = verify(SUMS, SIG, SETUP, &good, other_key).unwrap_err();
        assert!(e.contains("isn't signed"), "{e}");
    }

    #[test]
    fn placeholder_key_installs_nothing() {
        assert_eq!(RELEASE_KEY.trim(), "# no release key yet", "Task 6 replaces the placeholder");
        let good = sha256(FIXTURE_SETUP);
        assert_eq!(verify(SUMS, SIG, SETUP, &good, RELEASE_KEY), Err(NO_RELEASE_KEY.to_string()));
        assert_eq!(verify(SUMS, SIG, SETUP, &good, ""), Err(NO_RELEASE_KEY.to_string()));
        assert_eq!(NO_RELEASE_KEY, "this build has no release key");
    }

    /// The committed TEST-ONLY secret key (unencrypted: an empty password, never a prompt).
    fn test_secret_key() -> minisign::SecretKey {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/update/fixtures/TEST-ONLY.key");
        minisign::SecretKeyBox::from_string(&std::fs::read_to_string(path).unwrap()).unwrap().into_secret_key(Some(String::new())).unwrap()
    }

    /// `sums` signed on the spot with the TEST-ONLY key.
    fn signed(sums: &str) -> Vec<u8> {
        let sk = test_secret_key();
        let pk = minisign::PublicKey::from_secret_key(&sk).unwrap();
        minisign::sign(Some(&pk), &sk, std::io::Cursor::new(sums.as_bytes()), None, None).unwrap().into_string().into_bytes()
    }

    #[test]
    fn sums_entries_are_read_strictly() {
        let (a, b) = (sha256(b"a"), sha256(b"b"));
        let check = |sums: String, d: &[u8; 32]| verify(sums.as_bytes(), &signed(&sums), "x.exe", d, TEST_KEY);
        // CRLF lines (a sums file written on Windows), sha256sum's binary mode and upper-case hex read
        let crlf = format!("{}  y.exe\r\n{} *x.exe\r\n", hex(&b), hex(&a).to_uppercase());
        assert!(crlf.contains("\r\n"));
        assert_eq!(check(crlf, &a), Ok(()));
        assert_eq!(check(format!("{}  x.exe\r\n", hex(&a)), &a), Ok(()), "a CRLF line's name is not `x.exe\\r`");
        // listed twice with the same sum is fine; with another sum it is refused
        assert_eq!(check(format!("{0}  x.exe\n{0}  x.exe\n", hex(&a)), &a), Ok(()));
        let e = check(format!("{}  x.exe\n{}  x.exe\n", hex(&a), hex(&b)), &a).unwrap_err();
        assert!(e.contains("twice"), "{e}");
        // a name that only contains it, and a malformed sum, are not its entry
        assert!(check(format!("{}  x.exe.part\n{}  old-x.exe\n", hex(&a), hex(&a)), &a).unwrap_err().contains("isn't listed"));
        assert!(check(format!("{}  x.exe\n", &hex(&a)[..60]), &a).unwrap_err().contains("isn't a SHA-256"));
        // hex digits only: a `+` sign (which `from_str_radix` would take) or a non-hex letter is refused
        let plus = format!("+{}", &hex(&a)[1..]);
        assert!(check(format!("{plus}  x.exe\n"), &a).unwrap_err().contains("isn't a SHA-256"));
        assert_eq!(unhex(&format!("+a{}", &hex(&a)[2..])), None);
        assert_eq!(unhex(&format!("g{}", &hex(&a)[1..])), None);
        assert_eq!(unhex(&hex(&a)), Some(a));
        assert_eq!(unhex("zz"), None);
    }

    /// Regenerates the signed fixtures: `SHA256SUMS.txt` (from [`FIXTURE_SETUP`] and the zip
    /// stand-in) and its `.minisig`, signed with `fixtures/TEST-ONLY.key`; the key pair is generated
    /// first (no password, as `minisign -G -W`) when `TEST-ONLY.key` is missing. Run on purpose only:
    /// `cargo test -p app --release --bin app -- update::verify::tests::regenerate_signed_fixtures --ignored`.
    #[test]
    #[ignore = "writes the committed fixtures; run by hand"]
    fn regenerate_signed_fixtures() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join("update").join("fixtures");
        let (key_path, pub_path) = (dir.join("TEST-ONLY.key"), dir.join("TEST-ONLY.pub"));
        let sk = if key_path.exists() {
            minisign::SecretKeyBox::from_string(&std::fs::read_to_string(&key_path).unwrap()).unwrap().into_secret_key(Some(String::new())).unwrap()
        } else {
            // an empty password: checksummed, not encrypted (`generate_unencrypted_keypair` leaves
            // the checksum unset, so its box would not load back)
            let kp = minisign::KeyPair::generate_encrypted_keypair(Some(String::new())).unwrap();
            let pk_box = kp.pk.to_box().unwrap().into_string().replace("minisign public key", "TEST-ONLY minisign public key (signs nothing real)");
            std::fs::write(&pub_path, pk_box).unwrap();
            std::fs::write(&key_path, kp.sk.to_box(Some("TEST-ONLY minisign secret key (signs nothing real)")).unwrap().into_string()).unwrap();
            kp.sk
        };
        let pk = minisign::PublicKey::from_secret_key(&sk).unwrap();
        let sums = format!(
            "{}  AtomicSandbox-Setup-0.2.0.exe\n{}  AtomicSandbox-0.2.0-windows-x64.zip\n",
            hex(&sha256(FIXTURE_SETUP)),
            hex(&sha256(FIXTURE_ZIP))
        );
        std::fs::write(dir.join("SHA256SUMS.txt"), &sums).unwrap();
        let sig = minisign::sign(Some(&pk), &sk, std::io::Cursor::new(sums.as_bytes()), Some("TEST-ONLY fixture SHA256SUMS.txt"), Some("signature from the TEST-ONLY key"))
            .unwrap();
        std::fs::write(dir.join("SHA256SUMS.txt.minisig"), sig.into_string()).unwrap();
    }
}
