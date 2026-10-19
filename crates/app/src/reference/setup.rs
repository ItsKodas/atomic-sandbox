//! "Set up PySCF" (user request 2026-10-20: "make it a button"): installs a private Python with
//! PySCF for the reference comparison, step by step in child processes on a background thread, with
//! a live log and Cancel. Nothing runs until the user presses the button twice (the plan is shown
//! first).
//!
//! PySCF publishes wheels for Linux and macOS only (2.14.0 on PyPI: manylinux and macosx, no
//! win_amd64), so on **Windows** it lives in a WSL distro of its own, [`WSL_DISTRO`] (Ubuntu 24.04,
//! `wsl --install … --name … --no-launch`, WSL ≥ 2.4): the user's own distros are never touched, and
//! `wsl --unregister AtomicSandbox-PySCF` removes all of it. Inside it, as the distro's root:
//! `apt-get install python3-venv`, a venv at `/opt/pyscf-env`, `pip install pyscf geometric`.
//! **Elsewhere**: a venv in the user's data directory (`$XDG_DATA_HOME/atomic-sandbox/pyscf-env`,
//! else `~/.local/share/…`) made by `python3`, and the same pip install.
//!
//! Every command is an argument list (no shell, no quoting). The resulting interpreter is one of
//! [`super::candidates`]' fixed entries ([`managed_interpreter`]), so nothing needs saving: the probe
//! finds it next time.
use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The WSL distro the setup installs and uses (Windows).
pub const WSL_DISTRO: &str = "AtomicSandbox-PySCF";
/// The distro image it is installed from (`wsl --list --online`).
const WSL_IMAGE: &str = "Ubuntu-24.04";
/// The venv inside the WSL distro.
const WSL_ENV: &str = "/opt/pyscf-env";
const PACKAGES: [&str; 2] = ["pyscf", "geometric"];
/// Log lines kept for the panel.
const LOG_LINES: usize = 400;

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

/// The venv outside Windows.
fn native_env() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share")))?;
    Some(base.join("atomic-sandbox").join("pyscf-env"))
}

/// The interpreter the setup makes (tried first by the probe).
pub fn managed_interpreter() -> Option<Vec<String>> {
    if cfg!(windows) {
        Some(s(&["wsl", "-d", WSL_DISTRO, "-u", "root", "-e", &format!("{WSL_ENV}/bin/python")]))
    } else {
        native_env().map(|e| vec![e.join("bin").join("python").to_string_lossy().into_owned()])
    }
}

/// What pressing the button again will do (shown before it runs).
pub fn plan_text() -> String {
    if cfg!(windows) {
        format!(
            "PySCF has no Windows build, so this installs it in a private WSL distro: Ubuntu 24.04 named {WSL_DISTRO} (a download of a few hundred MB, ≈ 1–2 GB on disk; your own WSL distros are not touched), then Python's venv and `pip install pyscf geometric` inside it (≈ 300 MB). A few minutes on a fast connection. Remove it any time with `wsl --unregister {WSL_DISTRO}`."
        )
    } else {
        format!(
            "Makes a Python virtual environment at {} with `python3 -m venv` and runs `pip install pyscf geometric` in it (≈ 300 MB). Remove it by deleting that folder.",
            native_env().map_or("~/.local/share/atomic-sandbox/pyscf-env".into(), |p| p.display().to_string())
        )
    }
}

/// One step: what the log calls it, the command, how long it may take; `skip_if` = a command whose
/// success means the step is already done.
struct Step {
    label: &'static str,
    cmd: Vec<String>,
    timeout: Duration,
    skip_if: Option<Vec<String>>,
}

fn steps() -> Result<Vec<Step>, String> {
    let min = |m: u64| Duration::from_secs(60 * m);
    if cfg!(windows) {
        let root = |rest: &[&str]| {
            let mut v = s(&["wsl", "-d", WSL_DISTRO, "-u", "root", "-e"]);
            v.extend(s(rest));
            v
        };
        let py = format!("{WSL_ENV}/bin/python");
        let mut pip = root(&[&py, "-m", "pip", "install", "--progress-bar", "off"]);
        pip.extend(s(&PACKAGES));
        Ok(vec![
            Step {
                label: "install the WSL distro",
                cmd: s(&["wsl", "--install", WSL_IMAGE, "--name", WSL_DISTRO, "--no-launch"]),
                timeout: min(30),
                skip_if: Some(root(&["true"])),
            },
            Step { label: "update the package lists", cmd: root(&["apt-get", "update", "-q"]), timeout: min(10), skip_if: Some(root(&["test", "-x", &py])) },
            Step {
                label: "install Python's venv",
                cmd: root(&["env", "DEBIAN_FRONTEND=noninteractive", "apt-get", "install", "-y", "-q", "python3-venv"]),
                timeout: min(15),
                skip_if: Some(root(&["test", "-x", &py])),
            },
            Step { label: "make the environment", cmd: root(&["python3", "-m", "venv", WSL_ENV]), timeout: min(5), skip_if: Some(root(&["test", "-x", &py])) },
            Step { label: "pip install pyscf geometric", cmd: pip, timeout: min(30), skip_if: None },
        ])
    } else {
        let env = native_env().ok_or("no HOME or XDG_DATA_HOME to put the environment in")?;
        let py = env.join("bin").join("python").to_string_lossy().into_owned();
        let mut pip = vec![py.clone(), "-m".into(), "pip".into(), "install".into(), "--progress-bar".into(), "off".into()];
        pip.extend(s(&PACKAGES));
        Ok(vec![
            Step { label: "make the environment", cmd: vec!["python3".into(), "-m".into(), "venv".into(), env.to_string_lossy().into_owned()], timeout: min(5), skip_if: Some(vec![py.clone(), "-c".into(), "pass".into()]) },
            Step { label: "pip install pyscf geometric", cmd: pip, timeout: min(30), skip_if: None },
        ])
    }
}

fn command(cmd: &[String]) -> Command {
    let mut c = Command::new(&cmd[0]);
    c.args(&cmd[1..]).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        c.creation_flags(CREATE_NO_WINDOW);
    }
    c
}

/// Splits a child's output into log lines as it comes. wsl.exe writes its own messages in UTF-16:
/// the NULs are dropped, which leaves its (ASCII) text readable; `\r` progress redraws end a line.
fn pump(mut r: Box<dyn Read + Send>, log: Arc<Mutex<Vec<String>>>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut buf = [0u8; 4096];
        let mut line = Vec::new();
        let flush = |line: &mut Vec<u8>| {
            let t = String::from_utf8_lossy(line).trim().to_string();
            line.clear();
            if !t.is_empty() {
                let mut l = log.lock().unwrap();
                l.push(t);
                let n = l.len();
                if n > LOG_LINES {
                    l.drain(..n - LOG_LINES);
                }
            }
        };
        while let Ok(n) = r.read(&mut buf) {
            if n == 0 {
                break;
            }
            for &b in &buf[..n] {
                match b {
                    0 => {}
                    b'\n' | b'\r' => flush(&mut line),
                    b => line.push(b),
                }
            }
        }
        flush(&mut line);
    })
}

/// Runs `cmd` to its end (or `timeout`, or `cancel`), its output into `log`. `Ok(true)` = exit 0.
fn run(cmd: &[String], timeout: Duration, cancel: &AtomicBool, log: &Arc<Mutex<Vec<String>>>) -> Result<bool, String> {
    let mut child = command(cmd).spawn().map_err(|e| format!("{}: {e}", cmd.join(" ")))?;
    let out = pump(Box::new(child.stdout.take().expect("piped")), log.clone());
    let err = pump(Box::new(child.stderr.take().expect("piped")), log.clone());
    let start = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait().map_err(|e| e.to_string())? {
            break s;
        }
        if cancel.load(Ordering::Relaxed) || start.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(if cancel.load(Ordering::Relaxed) { "cancelled".into() } else { format!("timed out after {} min", timeout.as_secs() / 60) });
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let _ = (out.join(), err.join());
    Ok(status.success())
}

/// The setup in flight (or finished: `result`).
pub struct Setup {
    pub log: Arc<Mutex<Vec<String>>>,
    /// (step number from 1, of how many, its label).
    pub step: Arc<Mutex<(usize, usize, &'static str)>>,
    cancel: Arc<AtomicBool>,
    rx: Receiver<Result<(), String>>,
    pub started: Instant,
    pub result: Option<Result<(), String>>,
}

impl Setup {
    pub fn start() -> Setup {
        let log = Arc::new(Mutex::new(Vec::new()));
        let step = Arc::new(Mutex::new((0, 0, "starting")));
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = channel();
        let (l, st, c) = (log.clone(), step.clone(), cancel.clone());
        std::thread::Builder::new()
            .name("reference-setup".into())
            .spawn(move || {
                let r = (|| -> Result<(), String> {
                    let steps = steps()?;
                    let n = steps.len();
                    for (k, s) in steps.iter().enumerate() {
                        *st.lock().unwrap() = (k + 1, n, s.label);
                        if let Some(check) = &s.skip_if {
                            if run(check, Duration::from_secs(120), &c, &l)? {
                                l.lock().unwrap().push(format!("— {}: already done", s.label));
                                continue;
                            }
                        }
                        l.lock().unwrap().push(format!("— {} ({})", s.label, s.cmd.join(" ")));
                        if !run(&s.cmd, s.timeout, &c, &l)? {
                            return Err(format!("step {} of {n} failed: {} (the log above says why)", k + 1, s.label));
                        }
                    }
                    Ok(())
                })();
                let _ = tx.send(r);
            })
            .expect("spawn the setup thread");
        Setup { log, step, cancel, rx, started: Instant::now(), result: None }
    }

    pub fn running(&self) -> bool {
        self.result.is_none()
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Picks up the end (never blocks); true when it ended this call.
    pub fn poll(&mut self) -> bool {
        if self.result.is_some() {
            return false;
        }
        match self.rx.try_recv() {
            Ok(r) => self.result = Some(r),
            Err(TryRecvError::Empty) => return false,
            Err(TryRecvError::Disconnected) => self.result = Some(Err("the setup thread ended without a result".into())),
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The plan is argument lists only, ends with the pip install of both packages, and its
    /// interpreter is the one the probe tries first.
    #[test]
    fn the_plan_installs_into_the_interpreter_the_probe_tries_first() {
        let st = steps().unwrap();
        let last = &st.last().unwrap().cmd;
        assert!(last.ends_with(&s(&PACKAGES)) && last.iter().any(|a| a == "pip"), "{last:?}");
        let py = managed_interpreter().unwrap();
        assert!(last.windows(py.len()).any(|w| w == py.as_slice()) || last[0] == py[0], "pip runs in {py:?}: {last:?}");
        if std::env::var_os("ATOMIC_SANDBOX_PYTHON").is_none() {
            assert_eq!(super::super::candidates().first(), Some(&py));
        }
        assert!(!plan_text().is_empty());
    }

    /// Output split into lines: UTF-16 NULs dropped, `\r` and `\n` both end a line, blanks skipped.
    #[test]
    fn output_becomes_log_lines() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let bytes: Vec<u8> = "I\0n\0s\0t\0a\0l\0l\0\r\0\n\0".bytes().chain(b"Collecting pyscf\r50%\r100%\n\n".iter().copied()).collect();
        pump(Box::new(std::io::Cursor::new(bytes)), log.clone()).join().unwrap();
        assert_eq!(*log.lock().unwrap(), vec!["Install", "Collecting pyscf", "50%", "100%"]);
    }
}
