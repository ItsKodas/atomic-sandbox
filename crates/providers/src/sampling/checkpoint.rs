//! Pass checkpoints (user report 2026-10-20; the parent module doc, "Checkpoints"): a running
//! regular pass written as text after its solves, read back to go on from where it stood when the
//! app closed or crashed. Line-based, one record per line, numbers in Rust's shortest round-trip
//! form (`{:?}`), so a resumed pass holds bit for bit the geometries the interrupted one queued.
//!
//! What it holds: the pass's base scene and plan (as rescaled), its pair scans, its counters and
//! progress, the queued jobs with their geometries, the shell checks already solved (judged after
//! the pass's refit, so they must survive), the wall times behind the time limit, and the hot
//! burst — its positions, velocities, forces, evaluations left and restart counters. What it does
//! not hold: the samples (the surrogate's own file next to it — the header records how many there
//! were, and a resume with another count is refused), the reference cache's radial tables (the
//! references' energies are in that file; a later pass re-solves a table it needs), the notes
//! (drained after every solve), and the burst thermostat's random stream: `StdRng` has no stable
//! state to write, so a resumed burst draws from a fresh stream seeded from the pass seed and the
//! evaluations done — the same CSVR at the same temperature, another sequence of kicks (a sampler,
//! not a trajectory; module doc, 4.). A reference scan interrupted between multiplicities starts
//! over (its energies are set only when it completes). Teach-around and references-only passes
//! are not checkpointed: they are short and judged against the model as it stood.

use super::*;

/// The first line of a checkpoint.
pub const CHECKPOINT_MAGIC: &str = "atomic-sandbox pass checkpoint 1";

/// What a checkpoint says about itself without resuming it (the app's resume offer).
#[derive(Clone, Debug, PartialEq)]
pub struct CheckpointInfo {
    /// Jobs accounted for and planned (`SamplingProgress::done` / `total` as a resume starts).
    pub done: usize,
    pub total: usize,
    /// The stage of the next job.
    pub stage: &'static str,
    /// The pass's base scene (no velocities, no field).
    pub scene: Scene,
    /// `Provenance::describe` of the surface the pass teaches.
    pub provenance: String,
    /// Samples the surrogate held when it was written (the samples file next to it must hold as many).
    pub samples: usize,
    /// The store index of the pass's first sample (`SurrogateKs`'s pass coverage).
    pub pass_start: usize,
}

fn num(x: f64) -> String {
    format!("{x:?}")
}

fn nums(xs: impl IntoIterator<Item = f64>) -> String {
    xs.into_iter().map(num).collect::<Vec<_>>().join(" ")
}

fn flat(v: &[[f64; 3]]) -> String {
    nums(v.iter().flatten().copied())
}

fn opt(x: Option<f64>) -> String {
    x.map_or("none".into(), num)
}

fn words(v: &Option<Vec<String>>) -> Option<Vec<&str>> {
    v.as_ref().map(|v| v.iter().map(String::as_str).collect())
}

fn stage_word(w: &str) -> Option<&'static str> {
    ["perturbations", "scans", "teaching"].into_iter().find(|s| *s == w)
}

impl Sampler {
    /// The pass as a checkpoint (module doc): `samples` and `pass_start` are the surrogate's sample
    /// count and the pass's first store index, `provenance` the surface's `describe()`. `None` for a
    /// finished, teach-around or references-only pass.
    pub fn checkpoint(&self, provenance: &str, samples: usize, pass_start: usize) -> Option<String> {
        if self.progress.finished || self.is_targeted() || self.references_only {
            return None;
        }
        let mut o = String::new();
        let mut line = |s: String| {
            o.push_str(&s);
            o.push('\n');
        };
        line(CHECKPOINT_MAGIC.into());
        line(format!("provenance {}", provenance.replace('\n', " ")));
        line(format!("samples {samples} {pass_start}"));
        let sc = &self.scene;
        line(format!("cell {} {}", sc.cell.n, num(sc.cell.h)));
        line(format!("charge {} {} {} {}", sc.charge, sc.multiplicity.map_or("none".into(), |m| m.to_string()), opt(sc.sigma), sc.ionic.word()));
        for n in &sc.nuclei {
            line(format!("nucleus {} {} {} {}", n.z, n.a, num(n.mass), nums(n.pos)));
        }
        let p = &self.plan;
        line(format!(
            "plan {} {} {} {} {} {} {} {}",
            p.perturbations,
            num(p.perturbation_sigma),
            p.hot_burst_steps,
            num(p.hot_burst_t),
            p.scan_points_per_pair,
            num(p.scan_range.0),
            num(p.scan_range.1),
            p.seed
        ));
        line(format!("r_cut {}", num(self.r_cut)));
        line(format!("counters {} {} {} {} {} {}", self.refs_total, self.added, self.boxed, self.cell_skipped, self.tail_skipped, self.limit_skipped));
        line(format!("progress {} {} {}", self.progress.total, self.progress.skipped, self.progress.failed));
        for s in &self.scans {
            line(format!("scan {} {} {} {} {}", s.i, s.j, s.moved, num(s.r0), nums(s.r.iter().copied())));
        }
        line(format!("scan_planned {}", self.scan_planned.iter().map(usize::to_string).collect::<Vec<_>>().join(" ")));
        line(format!("walls {}", nums(self.solve_walls.iter().copied())));
        line(format!("floor {}", num(self.time_limit_floor_s)));
        let pos = |s: &Scene| flat(&s.nuclei.iter().map(|n| n.pos).collect::<Vec<_>>());
        for job in &self.queue {
            line(match job {
                Job::Reference(scan) => format!("job ref {}", scan.z()),
                Job::Perturbation(s) => format!("job pert {}", pos(s)),
                Job::Scan(s, pair, k) => format!("job scan {pair} {k} {}", pos(s)),
                Job::Shell(s, pair, k) => format!("job shell {pair} {k} {}", pos(s)),
                Job::Skip(stage, at, note) => {
                    let at = at.map_or("- -".into(), |(pair, k)| format!("{pair} {k}"));
                    match note {
                        Some(n) => format!("job skip {stage} {at} | {}", n.replace('\n', " ")),
                        None => format!("job skip {stage} {at}"),
                    }
                }
                Job::Target(_) => return None, // a regular pass has none
            });
        }
        let b = &self.burst;
        line(format!("burst {} {} {}", b.left, b.failures_in_a_row, b.restarts));
        line(format!("burst_pos {}", pos(&b.scene)));
        line(format!("burst_vel {}", flat(&b.scene.nuclei.iter().map(|n| n.vel).collect::<Vec<_>>())));
        if let Some(f) = &b.forces {
            line(format!("burst_forces {}", flat(f)));
        }
        for c in &self.shell_checks {
            line(format!(
                "shell_check {} {} {} {} {} {} {} {} | {}",
                c.pair.0,
                c.pair.1,
                num(c.r),
                num(c.r_cut),
                u8::from(c.unfit),
                opt(c.face),
                num(c.energy),
                pos(&c.scene),
                flat(&c.forces)
            ));
        }
        line("end".into());
        Some(o)
    }

    /// The pass a [`checkpoint`](Self::checkpoint) holds, ready to go on (module doc): the jobs it
    /// had queued, in order, with their geometries; the burst where it stood. Err for a text that
    /// is not a complete checkpoint of this format.
    pub fn resume(text: &str) -> Result<Sampler, String> {
        parse(text).map(|(s, _)| s)
    }
}

/// The header of a checkpoint (the app's offer), without building the pass's jobs for use.
pub fn checkpoint_info(text: &str) -> Result<CheckpointInfo, String> {
    parse(text).map(|(_, info)| info)
}

fn parse(text: &str) -> Result<(Sampler, CheckpointInfo), String> {
    let bad = |what: &str| format!("not a pass checkpoint this build can read ({what})");
    let mut lines = text.lines();
    if lines.next() != Some(CHECKPOINT_MAGIC) {
        return Err(bad("the first line"));
    }
    let f = |w: &str| w.parse::<f64>().map_err(|_| bad(&format!("number '{w}'")));
    let u = |w: &str| w.parse::<usize>().map_err(|_| bad(&format!("count '{w}'")));
    let of = |w: &str| if w == "none" { Ok(None) } else { f(w).map(Some) };
    let fs = |ws: &[&str]| ws.iter().map(|w| f(w)).collect::<Result<Vec<f64>, String>>();
    let triples = |ws: &[&str], n: usize| -> Result<Vec<[f64; 3]>, String> {
        let v = fs(ws)?;
        if v.len() != 3 * n {
            return Err(bad(&format!("{} coordinates for {n} nuclei", v.len())));
        }
        Ok(v.chunks(3).map(|c| [c[0], c[1], c[2]]).collect())
    };

    let mut provenance = None;
    let (mut samples, mut pass_start) = (None, 0);
    let mut cell = None;
    let mut settings = None;
    let mut nuclei = Vec::new();
    let mut plan = None;
    let mut r_cut = None;
    let mut counters = None;
    let mut progress = None;
    let mut scans = Vec::new();
    let mut scan_planned = Vec::new();
    let mut walls = Vec::new();
    let mut floor = TIME_LIMIT_FLOOR_S;
    let mut jobs: Vec<Vec<String>> = Vec::new();
    let mut burst = None;
    let (mut burst_pos, mut burst_vel, mut burst_forces) = (None, None, None);
    let mut shells: Vec<(Vec<String>, Vec<String>)> = Vec::new();
    let mut ended = false;
    for l in lines {
        let (head, tail) = match l.split_once(" | ") {
            Some((h, t)) => (h, Some(t)),
            None => (l, None),
        };
        let w: Vec<&str> = head.split(' ').filter(|w| !w.is_empty()).collect();
        let Some((&key, rest)) = w.split_first() else { continue };
        match key {
            "provenance" => provenance = Some(l["provenance ".len().min(l.len())..].to_string()),
            "samples" if rest.len() == 2 => {
                samples = Some(u(rest[0])?);
                pass_start = u(rest[1])?;
            }
            "cell" if rest.len() == 2 => cell = Some(sim::Cell { n: u(rest[0])?, h: f(rest[1])? }),
            "charge" if rest.len() == 4 => {
                let charge = rest[0].parse::<i32>().map_err(|_| bad("charge"))?;
                let mult = if rest[1] == "none" { None } else { Some(rest[1].parse::<u32>().map_err(|_| bad("multiplicity"))?) };
                let ionic = sim::IonicPolicy::parse(rest[3]).ok_or_else(|| bad("ionic policy"))?;
                settings = Some((charge, mult, of(rest[2])?, ionic));
            }
            "nucleus" if rest.len() == 6 => {
                let z = rest[0].parse::<u8>().map_err(|_| bad("Z"))?;
                let a = rest[1].parse::<u16>().map_err(|_| bad("A"))?;
                nuclei.push(sim::Nucleus { z, a, mass: f(rest[2])?, pos: [f(rest[3])?, f(rest[4])?, f(rest[5])?], vel: [0.0; 3] });
            }
            "plan" if rest.len() == 8 => {
                plan = Some(SamplingPlan {
                    perturbations: u(rest[0])?,
                    perturbation_sigma: f(rest[1])?,
                    hot_burst_steps: u(rest[2])?,
                    hot_burst_t: f(rest[3])?,
                    scan_points_per_pair: u(rest[4])?,
                    scan_range: (f(rest[5])?, f(rest[6])?),
                    seed: rest[7].parse::<u64>().map_err(|_| bad("seed"))?,
                })
            }
            "r_cut" if rest.len() == 1 => r_cut = Some(f(rest[0])?),
            "counters" if rest.len() == 6 => counters = Some(rest.iter().map(|w| u(w)).collect::<Result<Vec<_>, _>>()?),
            "progress" if rest.len() == 3 => progress = Some((u(rest[0])?, u(rest[1])?, u(rest[2])?)),
            "scan" if rest.len() >= 4 => scans.push(PairScan { i: u(rest[0])?, j: u(rest[1])?, moved: u(rest[2])?, r0: f(rest[3])?, r: fs(&rest[4..])? }),
            "scan_planned" => scan_planned = rest.iter().map(|w| u(w)).collect::<Result<_, _>>()?,
            "walls" => walls = fs(rest)?,
            "floor" if rest.len() == 1 => floor = f(rest[0])?,
            "job" if !rest.is_empty() => {
                let mut v: Vec<String> = rest.iter().map(|s| s.to_string()).collect();
                if let Some(t) = tail {
                    v.push("|".into());
                    v.push(t.to_string());
                }
                jobs.push(v);
            }
            "burst" if rest.len() == 3 => burst = Some((u(rest[0])?, u(rest[1])?, u(rest[2])?)),
            "burst_pos" => burst_pos = Some(rest.iter().map(|s| s.to_string()).collect::<Vec<_>>()),
            "burst_vel" => burst_vel = Some(rest.iter().map(|s| s.to_string()).collect::<Vec<_>>()),
            "burst_forces" => burst_forces = Some(rest.iter().map(|s| s.to_string()).collect::<Vec<_>>()),
            "shell_check" if tail.is_some() => {
                shells.push((rest.iter().map(|s| s.to_string()).collect(), tail.unwrap_or("").split(' ').filter(|w| !w.is_empty()).map(str::to_string).collect()))
            }
            "end" => {
                ended = true;
                break;
            }
            _ => return Err(bad(&format!("line '{}'", l.chars().take(40).collect::<String>()))),
        }
    }
    if !ended {
        return Err(bad("it ends early: written while the app stopped?"));
    }
    let need = |what: &str| bad(&format!("no {what}"));
    let cell = cell.ok_or_else(|| need("cell"))?;
    let (charge, multiplicity, sigma, ionic) = settings.ok_or_else(|| need("charge line"))?;
    if nuclei.is_empty() {
        return Err(need("nuclei"));
    }
    let scene = Scene { nuclei, cell, charge, multiplicity, sigma, external_field: [0.0; 3], ionic };
    let n = scene.nuclei.len();
    let plan = plan.ok_or_else(|| need("plan"))?;
    let c = counters.ok_or_else(|| need("counters"))?;
    let (total, skipped, failed) = progress.ok_or_else(|| need("progress"))?;
    if scan_planned.len() != scans.len() {
        return Err(bad("scan counts"));
    }
    let at = |s: &[[f64; 3]]| {
        let mut sc = scene.clone();
        for (nu, p) in sc.nuclei.iter_mut().zip(s) {
            nu.pos = *p;
        }
        sc
    };
    let mut queue = VecDeque::new();
    for j in &jobs {
        let w: Vec<&str> = j.iter().map(String::as_str).collect();
        let pair_k = |a: &str, b: &str| -> Result<(usize, usize), String> {
            let (pair, k) = (u(a)?, u(b)?);
            if pair >= scans.len() || k >= scans[pair].r.len() {
                return Err(bad("a scan job's indices"));
            }
            Ok((pair, k))
        };
        queue.push_back(match w.as_slice() {
            ["ref", z] => {
                let z = z.parse::<u8>().map_err(|_| bad("a reference's Z"))?;
                let nu = scene.nuclei.iter().find(|nu| nu.z == z).ok_or_else(|| bad("a reference of a species not in the scene"))?;
                Job::Reference(AtomScan::new(&scene.cell, scene.sigma, scene.ionic, z, nu.a, nu.mass))
            }
            ["pert", xs @ ..] => Job::Perturbation(at(&triples(xs, n)?)),
            ["scan", a, b, xs @ ..] => {
                let (pair, k) = pair_k(a, b)?;
                Job::Scan(at(&triples(xs, n)?), pair, k)
            }
            ["shell", a, b, xs @ ..] => {
                let (pair, k) = pair_k(a, b)?;
                Job::Shell(at(&triples(xs, n)?), pair, k)
            }
            ["skip", stage, a, b, more @ ..] => {
                let stage = stage_word(stage).ok_or_else(|| bad("a skip's stage"))?;
                let at = if *a == "-" { None } else { Some(pair_k(a, b)?) };
                let note = match more {
                    [] => None,
                    ["|", note] => Some(note.to_string()),
                    _ => return Err(bad("a skip's note")),
                };
                Job::Skip(stage, at, note)
            }
            _ => return Err(bad("a job")),
        });
    }
    let (left, failures_in_a_row, restarts) = burst.ok_or_else(|| need("burst"))?;
    let mut burst_scene = at(&triples(&words(&burst_pos).ok_or_else(|| need("burst positions"))?, n)?);
    for (nu, v) in burst_scene.nuclei.iter_mut().zip(triples(&words(&burst_vel).ok_or_else(|| need("burst velocities"))?, n)?) {
        nu.vel = v;
    }
    let forces = match words(&burst_forces) {
        Some(w) => Some(triples(&w, n)?),
        None => None,
    };
    // a fresh thermostat stream (module doc): seeded from the pass seed and the evaluations done
    let evaluated = plan.hot_burst_steps.saturating_sub(left) as u64;
    let seed = (plan.seed ^ 0x9E37_79B9_7F4A_7C15).wrapping_add(0xD1B5_4A32_D192_ED03u64.wrapping_mul(evaluated + 1));
    let mut shell_checks = Vec::new();
    for (head, tail) in &shells {
        let h: Vec<&str> = head.iter().map(String::as_str).collect();
        let [z0, z1, r, rc, unfit, face, energy, xs @ ..] = h.as_slice() else { return Err(bad("a shell check")) };
        let t: Vec<&str> = tail.iter().map(String::as_str).collect();
        shell_checks.push(ShellCheck {
            pair: (z0.parse().map_err(|_| bad("a shell check's Z"))?, z1.parse().map_err(|_| bad("a shell check's Z"))?),
            r: f(r)?,
            r_cut: f(rc)?,
            unfit: *unfit == "1",
            face: of(face)?,
            energy: f(energy)?,
            scene: at(&triples(xs, n)?),
            forces: triples(&t, n)?,
        });
    }
    let remaining = Sampler::jobs_in(&queue) + left;
    if remaining > total {
        return Err(bad("more jobs queued than planned"));
    }
    let stage = Sampler::stage_of(queue.front(), left);
    let info = CheckpointInfo {
        done: total - remaining,
        total,
        stage,
        scene: scene.clone(),
        provenance: provenance.ok_or_else(|| need("provenance"))?,
        samples: samples.ok_or_else(|| need("sample count"))?,
        pass_start,
    };
    let sampler = Sampler {
        scene: scene.clone(),
        plan: plan.clone(),
        refs_total: c[0],
        queue,
        scans,
        scan_planned,
        burst: Burst {
            left,
            scene: burst_scene,
            forces,
            csvr: Csvr::new(plan.hot_burst_t, HOT_BURST_TAU_FS, seed),
            dt: units::fs_to_au_time(HOT_BURST_DT_FS),
            trail: VecDeque::new(),
            failures_in_a_row,
            restarts,
        },
        refs: ReferenceCache::default(),
        progress: SamplingProgress { done: info.done, total, stage, last_energy: f64::NAN, cancelled: false, skipped, failed, finished: false, target: None },
        notes: Vec::new(),
        added: c[1],
        boxed: c[2],
        cell_skipped: c[3],
        tail_skipped: c[4],
        collection_skipped: 0,
        limit_skipped: c[5],
        last_failure: None,
        last_n: None,
        last_scan: None,
        interrupted: false,
        finished: None,
        targets: Vec::new(),
        holdouts: Vec::new(),
        shell_checks,
        r_cut: r_cut.ok_or_else(|| need("cutoff"))?,
        failed_reference: None,
        references_only: false,
        solve_walls: walls,
        timed_out: None,
        time_limit_floor_s: floor,
        timeouts_in_a_row: 0,
    };
    Ok((sampler, info))
}
