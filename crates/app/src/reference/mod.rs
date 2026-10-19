//! The optional reference comparison (user request 2026-10-20; notes "Queued 2026-10-20" →
//! "Reference comparison (PySCF, out of process)"): the current geometry answered by PySCF in a
//! **separate process** (`tools/reference/pyscf_ref.py`, embedded here and written to a fresh
//! private temporary directory per run), on a background thread with a timeout and a cancel. CLAUDE.md rule 2:
//! reference codes are oracles, never linked — nothing here feeds the dynamics (there is no
//! "reference" Play backend; that would be a rule change), and nothing runs unless the user presses
//! Compare. Python and PySCF are optional: without them the panel says how to enable it.
//!
//! What is compared (only quantities both sides define the same way — [`Comparison`]): the forces
//! per atom and their max deviation, the dipole (ours from the solved density,
//! `sim::charges::dipole`), and with "relax both" the bond lengths and angles of both relaxed
//! geometries. Raw total energies are shown side by side only for an all-electron scene, and never
//! as a deviation: ours carry the nuclear smearing σ and the grid; under pseudopotentials ours
//! exclude the cores entirely, and the panel says so instead.
//!
//! Interpreter: `ATOMIC_SANDBOX_PYTHON` (a path, or a command line such as `wsl -e python3`), else
//! `python`, `python3`, `py -3`, and on Windows `wsl -e python3` — PySCF publishes no Windows
//! build, so WSL is the usual way there. The first that answers `--probe` with PySCF is used.
pub mod json;

use json::Json;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The helper, embedded so the app needs no path to the repository.
const SCRIPT: &str = include_str!("../../../../tools/reference/pyscf_ref.py");
const SCRIPT_NAME: &str = "pyscf_ref.py";
/// The answer line's prefix (the script's `MARKER`).
const MARKER: &str = "@@PYSCF_REF@@";
/// How long one interpreter may take to answer `--probe` (WSL's first start takes seconds).
const PROBE_TIMEOUT: Duration = Duration::from_secs(40);
/// The default basis (def2 family: all-electron for Z ≤ 36).
pub const DEFAULT_BASIS: &str = "def2-TZVP";
/// What the panel says when no interpreter has PySCF.
pub const INSTALL_HINT: &str = "install Python and `pip install pyscf` to enable";

/// The reference method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Lda,
    Pbe,
    B3lyp,
    CcsdT,
}

impl Method {
    pub const ALL: [Method; 4] = [Method::Lda, Method::Pbe, Method::B3lyp, Method::CcsdT];

    /// The request's and the replay script's word.
    pub fn word(self) -> &'static str {
        match self {
            Method::Lda => "lda",
            Method::Pbe => "pbe",
            Method::B3lyp => "b3lyp",
            Method::CcsdT => "ccsd_t",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Method::Lda => "LDA (SPW92, this solver's functional)",
            Method::Pbe => "PBE",
            Method::B3lyp => "B3LYP",
            Method::CcsdT => "CCSD(T)",
        }
    }

    pub fn short(self) -> &'static str {
        match self {
            Method::Lda => "LDA",
            Method::Pbe => "PBE",
            Method::B3lyp => "B3LYP",
            Method::CcsdT => "CCSD(T)",
        }
    }

    /// The default timeout (minutes): CCSD(T) scales as N⁷.
    pub fn default_timeout_min(self) -> u32 {
        match self {
            Method::CcsdT => 30,
            _ => 10,
        }
    }
}

/// One geometry for the helper (atomic units).
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    pub z: Vec<u8>,
    pub positions: Vec<[f64; 3]>,
    pub charge: i32,
    pub multiplicity: Option<u32>,
    pub method: Method,
    pub basis: String,
    pub optimise: bool,
}

fn vec3s(v: &[[f64; 3]]) -> Json {
    Json::Arr(v.iter().map(|p| Json::Arr(p.iter().map(|&x| Json::Num(x)).collect())).collect())
}

impl Request {
    pub fn to_json(&self) -> Json {
        Json::Obj(vec![
            ("z".into(), Json::Arr(self.z.iter().map(|&z| Json::Num(z as f64)).collect())),
            ("positions".into(), vec3s(&self.positions)),
            ("charge".into(), Json::Num(self.charge as f64)),
            ("multiplicity".into(), self.multiplicity.map_or(Json::Null, |m| Json::Num(m as f64))),
            ("method".into(), Json::Str(self.method.word().into())),
            ("basis".into(), Json::Str(self.basis.clone())),
            ("optimise".into(), Json::Bool(self.optimise)),
        ])
    }
}

/// The helper's answer.
#[derive(Clone, Debug, PartialEq)]
pub struct Answer {
    pub energy: f64,
    pub forces: Option<Vec<[f64; 3]>>,
    pub dipole: Option<[f64; 3]>,
    pub optimised: Option<Vec<[f64; 3]>>,
    pub optimise_converged: Option<bool>,
    pub notes: Vec<String>,
    pub pyscf: String,
    pub wall_s: f64,
}

impl Answer {
    /// The answer object (`ok` true), checked against the request's atom count.
    pub fn from_json(v: &Json, atoms: usize) -> Result<Answer, String> {
        if v.get("ok").and_then(Json::bool) != Some(true) {
            return Err(v.get("error").and_then(Json::str).unwrap_or("the helper failed without a message").to_string());
        }
        let count = |what: &str, x: Option<Vec<[f64; 3]>>| -> Result<Option<Vec<[f64; 3]>>, String> {
            match x {
                Some(x) if x.len() != atoms => Err(format!("the helper's {what} have {} atoms, the request {atoms}", x.len())),
                x => Ok(x),
            }
        };
        let list = |key: &str| -> Result<Option<Vec<[f64; 3]>>, String> {
            match v.field(key) {
                None => Ok(None),
                Some(x) => x.vec3s().map(Some).ok_or(format!("the helper's {key} are not a list of 3-vectors")),
            }
        };
        Ok(Answer {
            energy: v.get("energy").and_then(Json::num).ok_or("the helper gave no energy")?,
            forces: count("forces", list("forces")?)?,
            dipole: match v.field("dipole") {
                None => None,
                Some(d) => Some(d.vec3().ok_or("the helper's dipole is not a 3-vector")?),
            },
            optimised: count("optimised positions", list("optimised")?)?,
            optimise_converged: v.field("optimise_converged").and_then(Json::bool),
            notes: v.get("notes").and_then(Json::arr).map(|a| a.iter().filter_map(|n| n.str().map(str::to_string)).collect()).unwrap_or_default(),
            pyscf: v.get("pyscf").and_then(Json::str).unwrap_or("?").to_string(),
            wall_s: v.get("wall_s").and_then(Json::num).unwrap_or(f64::NAN),
        })
    }
}

/// The answer line in the helper's stdout (the libraries' own output goes to stderr, but a C
/// library could still write to fd 1: only the marked line counts).
pub fn answer_line(stdout: &str) -> Option<Json> {
    stdout.lines().rev().find_map(|l| l.trim().strip_prefix(MARKER)).and_then(|t| Json::parse(t).ok())
}

// ---------------------------------------------------------------------------------------------
// The process

/// A fresh private directory holding only the script, for one run (removed when dropped). Never a
/// fixed path: `create_dir` fails on an existing one and the script is written `create_new`, so
/// nothing planted in the shared temporary directory (a `pyscf_ref.py`, or a module the
/// interpreter would import from the script's directory) can run in its place.
struct ScriptDir(PathBuf);

impl ScriptDir {
    fn new() -> Result<ScriptDir, String> {
        static RUN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.subsec_nanos());
        let dir = std::env::temp_dir().join(format!("atomic-sandbox-reference-{}-{nanos:09}-{}", std::process::id(), RUN.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        let guard = ScriptDir(dir);
        let path = guard.0.join(SCRIPT_NAME);
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .and_then(|mut f| f.write_all(SCRIPT.as_bytes()))
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        Ok(guard)
    }
}

impl Drop for ScriptDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(self.0.join(SCRIPT_NAME));
        let _ = std::fs::remove_dir(&self.0);
    }
}

/// The interpreters to try, in order (module doc).
pub fn candidates() -> Vec<Vec<String>> {
    let mut out = Vec::new();
    if let Ok(v) = std::env::var("ATOMIC_SANDBOX_PYTHON") {
        let v = v.trim().trim_matches('"').to_string();
        if !v.is_empty() {
            out.push(if Path::new(&v).is_file() { vec![v] } else { v.split_whitespace().map(str::to_string).collect() });
        }
    }
    let words: &[&[&str]] = if cfg!(windows) { &[&["python"], &["python3"], &["py", "-3"], &["wsl", "-e", "python3"]] } else { &[&["python3"], &["python"]] };
    out.extend(words.iter().map(|w| w.iter().map(|s| s.to_string()).collect()));
    out
}

pub fn command_line(cmd: &[String]) -> String {
    cmd.join(" ")
}

/// Runs `cmd pyscf_ref.py args…` in the script's directory with `stdin`, until it exits, `timeout`
/// passes or `cancel` is set (then the process is killed). Returns the answer line's JSON, or why
/// there is none (with the end of stderr).
fn run_helper(cmd: &[String], args: &[&str], stdin: &str, timeout: Duration, cancel: &AtomicBool) -> Result<Json, String> {
    let dir = ScriptDir::new()?;
    let mut c = Command::new(&cmd[0]);
    c.args(&cmd[1..]).arg(SCRIPT_NAME).args(args).current_dir(&dir.0).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(windows)]
    {
        // the app is a windowed program: a console child would open a console window
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        c.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = c.spawn().map_err(|e| format!("{}: {e}", command_line(cmd)))?;
    let mut input = child.stdin.take().expect("piped");
    let text = stdin.to_string();
    std::thread::spawn(move || {
        let _ = input.write_all(text.as_bytes());
    });
    let reader = |mut r: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut s = Vec::new();
            let _ = r.read_to_end(&mut s);
            String::from_utf8_lossy(&s).into_owned()
        })
    };
    let out = reader(Box::new(child.stdout.take().expect("piped")));
    let err = reader(Box::new(child.stderr.take().expect("piped")));
    let start = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break s,
            Ok(None) => {}
            Err(e) => return Err(format!("waiting for the helper: {e}")),
        }
        let stop = if cancel.load(Ordering::Relaxed) {
            Some("cancelled".to_string())
        } else if start.elapsed() > timeout {
            Some(format!("timed out after {} s", timeout.as_secs()))
        } else {
            None
        };
        if let Some(why) = stop {
            let _ = child.kill();
            let _ = child.wait();
            return Err(why);
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let (out, err) = (out.join().unwrap_or_default(), err.join().unwrap_or_default());
    match answer_line(&out) {
        Some(v) => Ok(v),
        None => {
            // wsl.exe writes its own errors in UTF-16: drop the NULs so the message reads
            let tail: String = err.replace('\0', "").trim().chars().rev().take(400).collect::<Vec<_>>().into_iter().rev().collect();
            Err(format!("{} exited ({status}) without an answer{}", command_line(cmd), if tail.is_empty() { String::new() } else { format!(": {tail}") }))
        }
    }
}

/// An interpreter with PySCF.
#[derive(Clone, Debug, PartialEq)]
pub struct Interpreter {
    pub cmd: Vec<String>,
    pub pyscf: String,
    pub python: String,
    /// The geometry optimiser PySCF can drive (`geometric`, `pyberny`), if installed.
    pub geomopt: Option<String>,
}

/// The probe's state.
pub enum Probe {
    NotRun,
    Running(Receiver<Result<Interpreter, String>>),
    Ready(Interpreter),
    /// No interpreter has PySCF: what each candidate said.
    Missing(String),
}

fn probe_all(cancel: &AtomicBool) -> Result<Interpreter, String> {
    let mut why = Vec::new();
    for cmd in candidates() {
        match run_helper(&cmd, &["--probe"], "", PROBE_TIMEOUT, cancel) {
            Ok(v) if v.get("ok").and_then(Json::bool) == Some(true) => {
                return Ok(Interpreter {
                    pyscf: v.get("pyscf").and_then(Json::str).unwrap_or("?").into(),
                    python: v.get("python").and_then(Json::str).unwrap_or("?").into(),
                    geomopt: v.field("geomopt").and_then(Json::str).map(str::to_string),
                    cmd,
                })
            }
            Ok(v) => why.push(format!("{}: {}", command_line(&cmd), v.get("error").and_then(Json::str).unwrap_or("no PySCF"))),
            Err(e) => why.push(e),
        }
    }
    Err(why.join("\n"))
}

/// A helper run in flight.
pub struct Job {
    cancel: Arc<AtomicBool>,
    rx: Receiver<Result<Answer, String>>,
    pub started: Instant,
    pub timeout: Duration,
}

impl Job {
    fn spawn(interp: &Interpreter, req: &Request, timeout: Duration) -> Job {
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = channel();
        let (cmd, text, atoms, flag) = (interp.cmd.clone(), req.to_json().write(), req.z.len(), cancel.clone());
        std::thread::Builder::new()
            .name("reference".into())
            .spawn(move || {
                let r = run_helper(&cmd, &[], &text, timeout, &flag).and_then(|v| Answer::from_json(&v, atoms));
                let _ = tx.send(r);
            })
            .expect("spawn the reference thread");
        Job { cancel, rx, started: Instant::now(), timeout }
    }
}

// ---------------------------------------------------------------------------------------------
// Ours, and the comparison

/// Our side, captured when Compare was pressed (and filled in as the snapshots bring the rest).
#[derive(Clone, Debug)]
pub struct Ours {
    pub z: Vec<u8>,
    pub positions: Vec<[f64; 3]>,
    /// The provider's raw forces at `positions` (Ha/Bohr).
    pub forces: Vec<[f64; 3]>,
    pub energy: f64,
    /// The dipole of the density solved at `positions` (`PartialCharges::dipole`), once a snapshot
    /// carries charges measured exactly there.
    pub dipole: Option<[f64; 3]>,
    /// Something is pseudised: our E excludes the cores.
    pub pseudised: bool,
    pub estimated: bool,
    pub engine: String,
    pub sigma: f64,
    pub grid: (usize, f64),
    pub charge: i32,
    /// "Relax both": our relaxed positions once our relax ended, and how it ended.
    pub relaxed: Option<Vec<[f64; 3]>>,
    pub relax_note: Option<String>,
    /// The bonds the geometry tables use (our bond graph's pairs, else every pair — [`pairs_for`]).
    pub bonds: Vec<(usize, usize)>,
    /// What constrains our relax and not theirs (walls, pins, an external field), for the note.
    pub constraints: Vec<&'static str>,
    pub bonds_from_graph: bool,
}

/// "Relax both": where our relax is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelaxPhase {
    /// The command is sent; the snapshots still show the previous relax (or none).
    Waiting,
    Running,
    Done,
}

/// One comparison: the request, our side, the job, their answer.
pub struct Run {
    pub request: Request,
    pub ours: Ours,
    pub job: Option<Job>,
    pub answer: Option<Result<Answer, String>>,
    pub elapsed: Option<Duration>,
    pub relax: Option<RelaxPhase>,
    /// The snapshot's relax before ours was sent (to tell ours from an earlier one).
    pub relax_before: Option<sim::RelaxProgress>,
    pub sent: Instant,
}

impl Run {
    pub fn running(&self) -> bool {
        self.job.is_some() || matches!(self.relax, Some(RelaxPhase::Waiting | RelaxPhase::Running))
    }
}

/// The panel's state (in `UiState`).
pub struct Compare {
    pub method: Method,
    pub relax: bool,
    pub timeout_min: u32,
    pub probe: Probe,
    pub run: Option<Run>,
    /// Draw their forces as arrows in the view (display only).
    pub arrows: bool,
    /// Physics ▸ Compare with reference… asked for the section open.
    pub open_request: bool,
}

impl Default for Compare {
    fn default() -> Compare {
        Compare { method: Method::Lda, relax: false, timeout_min: Method::Lda.default_timeout_min(), probe: Probe::NotRun, run: None, arrows: true, open_request: false }
    }
}

impl Compare {
    /// Starts the probe (once; again after "check again").
    pub fn ensure_probe(&mut self) {
        if matches!(self.probe, Probe::NotRun) {
            let (tx, rx) = channel();
            std::thread::Builder::new()
                .name("reference-probe".into())
                .spawn(move || {
                    let _ = tx.send(probe_all(&AtomicBool::new(false)));
                })
                .expect("spawn the probe thread");
            self.probe = Probe::Running(rx);
        }
    }

    pub fn interpreter(&self) -> Option<&Interpreter> {
        match &self.probe {
            Probe::Ready(i) => Some(i),
            _ => None,
        }
    }

    pub fn busy(&self) -> bool {
        self.run.as_ref().is_some_and(Run::running)
    }

    /// Starts a run for `ours` (the caller checked the snapshot). `Err` says why not.
    pub fn start(&mut self, ours: Ours, method: Method, relax: bool, multiplicity: Option<u32>, relax_before: Option<sim::RelaxProgress>) -> Result<(), String> {
        let interp = self.interpreter().ok_or_else(|| format!("no Python with PySCF: {INSTALL_HINT}"))?.clone();
        if self.busy() {
            return Err("a comparison is running: cancel it first".into());
        }
        if relax && interp.geomopt.is_none() {
            return Err("relaxing the reference needs the geomeTRIC package: `pip install geometric` (or pyberny)".into());
        }
        let request = Request { z: ours.z.clone(), positions: ours.positions.clone(), charge: ours.charge, multiplicity, method, basis: DEFAULT_BASIS.into(), optimise: relax };
        let job = Job::spawn(&interp, &request, Duration::from_secs(self.timeout_min.max(1) as u64 * 60));
        self.run = Some(Run { request, ours, job: Some(job), answer: None, elapsed: None, relax: relax.then_some(RelaxPhase::Waiting), relax_before, sent: Instant::now() });
        Ok(())
    }

    pub fn cancel(&mut self) {
        if let Some(job) = self.run.as_ref().and_then(|r| r.job.as_ref()) {
            job.cancel.store(true, Ordering::Relaxed);
        }
    }

    /// Once per frame: the probe's and the job's results, our dipole and our relax from the
    /// snapshot. Never blocks.
    pub fn poll(&mut self, snap: Option<&crate::sim::Snapshot>) {
        if let Probe::Running(rx) = &self.probe {
            match rx.try_recv() {
                Ok(Ok(i)) => self.probe = Probe::Ready(i),
                Ok(Err(why)) => self.probe = Probe::Missing(why),
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => self.probe = Probe::Missing("the probe thread ended".into()),
            }
        }
        let Some(run) = self.run.as_mut() else { return };
        if let Some(job) = &run.job {
            match job.rx.try_recv() {
                Ok(r) => {
                    run.elapsed = Some(job.started.elapsed());
                    run.answer = Some(r);
                    run.job = None;
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    run.answer = Some(Err("the reference thread ended without an answer".into()));
                    run.job = None;
                }
            }
        }
        let Some(s) = snap else { return };
        let d = &s.dynamics;
        // our dipole: charges measured exactly at the captured geometry (before a relax moves it)
        if run.ours.dipole.is_none() {
            if let Some(c) = d.charges.as_ref().filter(|c| c.balanced() && c.z == run.ours.z && c.positions == run.ours.positions) {
                run.ours.dipole = Some(c.dipole);
            }
        }
        match run.relax {
            Some(RelaxPhase::Waiting) => {
                if d.relax.as_ref().is_some_and(|r| r.converged.is_none()) {
                    run.relax = Some(RelaxPhase::Running);
                } else if d.relax.is_some() && d.relax != run.relax_before && run.sent.elapsed() > Duration::from_millis(500) {
                    // a relax that started and ended between two snapshots
                    run.relax = Some(RelaxPhase::Running);
                } else if d.last_rejection.is_some() && run.sent.elapsed() > Duration::from_secs(2) && d.relax == run.relax_before {
                    run.relax = Some(RelaxPhase::Done);
                    run.ours.relax_note = Some(format!("our relax was refused: {}", d.last_rejection.as_deref().unwrap_or("")));
                }
            }
            Some(RelaxPhase::Running) => {
                if let Some(r) = d.relax.as_ref().filter(|r| r.converged.is_some()) {
                    if !d.forces_stale && d.scene.nuclei.len() == run.ours.z.len() {
                        run.relax = Some(RelaxPhase::Done);
                        run.ours.relaxed = Some(d.scene.nuclei.iter().map(|n| n.pos).collect());
                        run.ours.relax_note = Some(match (r.converged, &r.note) {
                            (Some(true), _) => format!("ours converged in {} steps (max |F| {:.1e} Ha/Bohr)", r.steps, r.max_force),
                            (_, Some(n)) => format!("ours ended unconverged: {n}"),
                            _ => "ours ended unconverged".into(),
                        });
                        if let Some(g) = d.bonds.as_ref().filter(|g| g.n_nuclei == run.ours.z.len()) {
                            run.ours.bonds = g.bonds.iter().map(|b| (b.i.min(b.j), b.i.max(b.j))).collect();
                            run.ours.bonds_from_graph = true;
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// The bonds the geometry tables measure: our bond graph's pairs when it has some, else every pair
/// (small scenes; display only — no per-element radius decides it, rule 1).
pub fn pairs_for(n: usize, graph: Option<&sim::BondGraph>) -> (Vec<(usize, usize)>, bool) {
    if let Some(g) = graph.filter(|g| g.n_nuclei == n && !g.bonds.is_empty()) {
        let mut b: Vec<_> = g.bonds.iter().map(|b| (b.i.min(b.j), b.i.max(b.j))).collect();
        b.sort_unstable();
        b.dedup();
        return (b, true);
    }
    ((0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j))).collect(), false)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// Per-atom |F_ours − F_ref|, their max and RMS (Ha/Bohr).
#[derive(Clone, Debug, PartialEq)]
pub struct ForceDeviation {
    pub per_atom: Vec<f64>,
    pub max: f64,
    pub rms: f64,
    /// Which atom has the max.
    pub worst: usize,
}

pub fn force_deviation(ours: &[[f64; 3]], theirs: &[[f64; 3]]) -> Option<ForceDeviation> {
    if ours.len() != theirs.len() || ours.is_empty() {
        return None;
    }
    let per_atom: Vec<f64> = ours.iter().zip(theirs).map(|(a, b)| norm(sub(*a, *b))).collect();
    let (worst, max) = per_atom.iter().copied().enumerate().fold((0, 0.0), |m, (i, x)| if x > m.1 { (i, x) } else { m });
    let rms = (per_atom.iter().map(|x| x * x).sum::<f64>() / per_atom.len() as f64).sqrt();
    Some(ForceDeviation { per_atom, max, rms, worst })
}

/// The bond lengths (Bohr) of `pos` for `bonds`.
pub fn bond_lengths(pos: &[[f64; 3]], bonds: &[(usize, usize)]) -> Vec<f64> {
    bonds.iter().map(|&(i, j)| norm(sub(pos[i], pos[j]))).collect()
}

/// Every angle i–j–k at a centre j between two of its bonds (i < k), in a fixed order.
pub fn angle_triples(n: usize, bonds: &[(usize, usize)]) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    for j in 0..n {
        let mut nb: Vec<usize> = bonds.iter().filter_map(|&(a, b)| if a == j { Some(b) } else if b == j { Some(a) } else { None }).collect();
        nb.sort_unstable();
        nb.dedup();
        for (x, &i) in nb.iter().enumerate() {
            for &k in &nb[x + 1..] {
                out.push((i, j, k));
            }
        }
    }
    out
}

/// The angle i–j–k (degrees).
pub fn angle(pos: &[[f64; 3]], (i, j, k): (usize, usize, usize)) -> f64 {
    let (a, b) = (sub(pos[i], pos[j]), sub(pos[k], pos[j]));
    let (na, nb) = (norm(a), norm(b));
    if na == 0.0 || nb == 0.0 {
        return f64::NAN; // two nuclei on one point: no angle
    }
    let c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb);
    c.clamp(-1.0, 1.0).acos().to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_request_writes_what_the_helper_reads() {
        let r = Request { z: vec![8, 1, 1], positions: vec![[0.0, 0.0, 0.2], [1.43, 0.0, -0.9], [-1.43, 0.0, -0.9]], charge: 0, multiplicity: None, method: Method::CcsdT, basis: DEFAULT_BASIS.into(), optimise: true };
        let v = Json::parse(&r.to_json().write()).unwrap();
        assert_eq!(v.get("z").and_then(Json::arr).map(|a| a.len()), Some(3));
        assert_eq!(v.get("positions").and_then(Json::vec3s), Some(r.positions.clone()));
        assert_eq!(v.get("method").and_then(Json::str), Some("ccsd_t"));
        assert_eq!(v.get("multiplicity"), Some(&Json::Null));
        assert_eq!(v.get("optimise").and_then(Json::bool), Some(true));
    }

    #[test]
    fn the_answer_line_is_found_among_library_chatter() {
        let out = "converged SCF energy = -1.13\n@@PYSCF_REF@@{\"ok\": true, \"energy\": -1.1372, \"forces\": [[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]], \"dipole\": [0.0, 0.0, 0.0], \"optimised\": null, \"notes\": [\"restricted\"], \"pyscf\": \"2.14.0\", \"wall_s\": 1.5}\n";
        let a = Answer::from_json(&answer_line(out).unwrap(), 2).unwrap();
        assert_eq!(a.energy, -1.1372);
        assert_eq!(a.forces.as_deref(), Some(&[[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]][..]));
        assert_eq!((a.dipole, a.optimised.is_none(), a.notes.len(), a.pyscf.as_str()), (Some([0.0; 3]), true, 1, "2.14.0"));
        // the atom count is checked, and a failure carries its message
        assert!(Answer::from_json(&answer_line(out).unwrap(), 3).is_err());
        let bad = answer_line("@@PYSCF_REF@@{\"ok\": false, \"error\": \"ModuleNotFoundError: No module named 'pyscf'\"}").unwrap();
        assert_eq!(Answer::from_json(&bad, 2).unwrap_err(), "ModuleNotFoundError: No module named 'pyscf'");
        assert!(answer_line("no marker here\n").is_none());
    }

    #[test]
    fn the_force_deviation_is_per_atom_with_its_max_and_rms() {
        let ours = [[0.0, 0.0, 0.03], [0.0, 0.0, -0.03], [0.01, 0.0, 0.0]];
        let theirs = [[0.0, 0.0, 0.0], [0.0, 0.04, -0.03], [0.01, 0.0, 0.0]];
        let d = force_deviation(&ours, &theirs).unwrap();
        assert_eq!(d.per_atom, vec![0.03, 0.04, 0.0]);
        assert_eq!((d.max, d.worst), (0.04, 1));
        assert!((d.rms - ((0.03f64.powi(2) + 0.04f64.powi(2)) / 3.0).sqrt()).abs() < 1e-15);
        assert!(force_deviation(&ours, &theirs[..2]).is_none());
    }

    #[test]
    fn lengths_and_angles_of_a_bent_triatomic() {
        // water-like: O at the origin, the H's at 1.8 Bohr and 104.5° apart
        let t = 104.5f64.to_radians() / 2.0;
        let pos = [[0.0; 3], [1.8 * t.sin(), 0.0, 1.8 * t.cos()], [-1.8 * t.sin(), 0.0, 1.8 * t.cos()]];
        let bonds = [(0, 1), (0, 2)];
        let l = bond_lengths(&pos, &bonds);
        assert!(l.iter().all(|x| (x - 1.8).abs() < 1e-12), "{l:?}");
        let tri = angle_triples(3, &bonds);
        assert_eq!(tri, vec![(1, 0, 2)]);
        assert!((angle(&pos, tri[0]) - 104.5).abs() < 1e-10);
        // no graph: every pair, so the H–H distance too, and three angles
        let (pairs, from_graph) = pairs_for(3, None);
        assert_eq!((pairs, from_graph), (vec![(0, 1), (0, 2), (1, 2)], false));
        assert_eq!(angle_triples(3, &[(0, 1), (0, 2), (1, 2)]).len(), 3);
    }
}
