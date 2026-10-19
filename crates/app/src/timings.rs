//! Completion times (user request 2026-10-19): how long operations took — a scene's first solve
//! after loading it, keyed by the scene, its grid, its ionic potential and the solver — for the
//! Activity tab's ETAs. The user's own times live in `%LOCALAPPDATA%\atomic-sandbox\timings.txt`
//! (an average per key, updated as they happen); times shipped with the repository
//! (`timings/timings.txt`, `scene::timings_dir`) fill in what the user has not measured yet. The
//! Profiles window copies the user's into the repository file, to commit. One `key = seconds count`
//! line per entry. Display only: nothing here reaches the solver.
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Default)]
pub struct Timings {
    /// The user's own: key → (mean seconds, how many).
    mine: BTreeMap<String, (f64, u32)>,
    /// Shipped with the repository.
    shipped: BTreeMap<String, (f64, u32)>,
    mine_file: Option<PathBuf>,
}

fn parse(text: &str) -> BTreeMap<String, (f64, u32)> {
    text.lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .filter_map(|l| {
            let (k, v) = l.split_once('=')?;
            let mut it = v.split_whitespace();
            let secs: f64 = it.next()?.parse().ok()?;
            let n: u32 = it.next().and_then(|x| x.parse().ok()).unwrap_or(1);
            (secs.is_finite() && secs >= 0.0).then(|| (k.trim().to_string(), (secs, n.max(1))))
        })
        .collect()
}

fn text(map: &BTreeMap<String, (f64, u32)>) -> String {
    let mut out = String::from("# Atomic Sandbox completion times: key = mean seconds, how many\n");
    for (k, (s, n)) in map {
        out += &format!("{k} = {s:.3} {n}\n");
    }
    out
}

impl Timings {
    /// The user's file (the window) and the repository's.
    pub fn load(mine_file: Option<PathBuf>) -> Timings {
        let read = |p: &PathBuf| std::fs::read_to_string(p).map(|t| parse(&t)).unwrap_or_default();
        let shipped = crate::scene::timings_dir().map(|d| read(&d.join("timings.txt"))).unwrap_or_default();
        let mine = mine_file.as_ref().map(read).unwrap_or_default();
        Timings { mine, shipped, mine_file }
    }

    /// The expected seconds for `key`: the user's own, else the repository's.
    pub fn eta(&self, key: &str) -> Option<f64> {
        self.mine.get(key).or_else(|| self.shipped.get(key)).map(|e| e.0)
    }

    /// One more completion of `key` in `secs` (a running mean over the last ~10), written at once.
    pub fn record(&mut self, key: &str, secs: f64) {
        if !(secs.is_finite() && secs > 0.0) {
            return;
        }
        let e = self.mine.entry(key.to_string()).or_insert((secs, 0));
        let n = (e.1 + 1).min(10);
        e.0 += (secs - e.0) / n as f64;
        e.1 = e.1.saturating_add(1);
        if let Some(p) = &self.mine_file {
            if let Some(dir) = p.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(p, text(&self.mine));
        }
    }

    /// (the user's entries, the shipped ones).
    pub fn counts(&self) -> (usize, usize) {
        (self.mine.len(), self.shipped.len())
    }

    /// The user's times merged into the repository's file (`timings/timings.txt`, created when
    /// missing), the user's winning where both have a key — to commit and share.
    pub fn publish(&mut self) -> Result<PathBuf, String> {
        let dir = crate::scene::timings_dir().unwrap_or_else(|| PathBuf::from("timings"));
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let mut merged = self.shipped.clone();
        merged.extend(self.mine.iter().map(|(k, v)| (k.clone(), *v)));
        let path = dir.join("timings.txt");
        std::fs::write(&path, text(&merged)).map_err(|e| e.to_string())?;
        self.shipped = merged;
        Ok(path)
    }
}
