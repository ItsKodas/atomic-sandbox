//! Scene text format v1 (see module tests for the grammar) and embedded presets.
//!
//! One directive per line, `#` starts a comment, fields are whitespace-separated:
//! ```text
//! # atomic sandbox scene v1
//! grid 49 0.25            # n h   (cubic, Bohr)
//! charge 0
//! multiplicity 2          # optional
//! sigma 0.25              # optional; default = h
//! nucleus 1  0.7 0 0      # Z x y z [vx vy vz]        — the most abundant isotope
//! nucleus 1 2 -0.7 0 0    # Z A x y z [vx vy vz]      — isotope A (M6 spec §5.1)
//! ionic cores             # optional (M7): none | cores | all — how the nuclei and their cores
//!                         # enter the solve; a file without the line means `none`, the all-electron
//!                         # physics it was written for (spec §4.7)
//! ```
//! Positions are in Bohr, velocities in Bohr per atomic time unit. The two `nucleus` shapes are
//! told apart by their field count (5 or 8: no A; 6 or 9: A). The writer emits A only when it
//! is not the most abundant isotope, so a file without isotopes round-trips byte-identically.

#[derive(Clone, Debug, PartialEq)]
pub struct SceneNucleus {
    pub z: u8,
    /// The mass number; `None` = the most abundant isotope (`units::most_abundant_a`). The
    /// parser and `from_sim` normalise an explicit most-abundant A to `None`, so equality is
    /// insensitive to how it was spelled.
    pub a: Option<u16>,
    pub pos: [f64; 3],
    pub vel: [f64; 3],
}

impl SceneNucleus {
    /// The mass number shown for this nucleus: its own `a`, else the most abundant isotope's;
    /// `None` beyond the table.
    pub fn mass_number(&self) -> Option<u16> {
        self.a.or_else(|| units::most_abundant_a(self.z))
    }
}

/// `Some(a)` unless `a` is Z's most abundant isotope (the canonical `SceneNucleus::a`).
pub(crate) fn canonical_a(z: u8, a: u16) -> Option<u16> {
    if units::most_abundant_a(z) == Some(a) { None } else { Some(a) }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Scene {
    pub nuclei: Vec<SceneNucleus>,
    pub charge: i32,
    pub multiplicity: Option<u32>,
    pub grid_n: usize,
    pub grid_h: f64,
    pub sigma: Option<f64>,
    /// The ionic policy (M7, `sim::IonicPolicy`): a file without an `ionic` line is `None`, and
    /// [`Scene::preset`] is the file as written (so `None`); the app applies
    /// [`crate::ui::DEFAULT_IONIC`] to the presets it starts with.
    pub ionic: sim::IonicPolicy,
}

impl Scene {
    /// The preset names in the popup's order: `empty` (the default cell with no nuclei — nothing
    /// to solve until atoms are placed) first, then the molecules.
    pub const PRESETS: &'static [&'static str] = &["empty", "h", "h2", "he", "lih", "h2o"];

    pub fn preset(name: &str) -> Option<Scene> {
        let text = match name {
            "empty" => include_str!("../presets/empty.scene"),
            "h" => include_str!("../presets/h.scene"),
            "h2" => include_str!("../presets/h2.scene"),
            "he" => include_str!("../presets/he.scene"),
            "lih" => include_str!("../presets/lih.scene"),
            "h2o" => include_str!("../presets/h2o.scene"),
            _ => return None,
        };
        Some(Scene::parse(text).expect("preset parses"))
    }

    pub fn parse(text: &str) -> Result<Scene, String> {
        let (mut grid, mut charge, mut mult, mut sigma, mut nuclei) =
            (None, 0i32, None, None, Vec::new());
        let mut ionic = sim::IonicPolicy::None;
        for (i, raw) in text.lines().enumerate() {
            let line = raw.split('#').next().unwrap().trim();
            if line.is_empty() {
                continue;
            }
            let t: Vec<&str> = line.split_whitespace().collect();
            let num = |s: &str| {
                s.parse::<f64>()
                    .map_err(|_| format!("line {}: bad number '{s}'", i + 1))
            };
            match t[0] {
                "grid" if t.len() == 3 => {
                    let n = t[1]
                        .parse::<usize>()
                        .map_err(|_| format!("line {}: bad n", i + 1))?;
                    grid = Some((n, num(t[2])?));
                }
                "charge" if t.len() == 2 => {
                    charge = t[1]
                        .parse()
                        .map_err(|_| format!("line {}: bad charge", i + 1))?;
                }
                "multiplicity" if t.len() == 2 => {
                    mult = Some(
                        t[1].parse()
                            .map_err(|_| format!("line {}: bad multiplicity", i + 1))?,
                    );
                }
                "sigma" if t.len() == 2 => sigma = Some(num(t[1])?),
                "ionic" if t.len() == 2 => {
                    ionic = sim::IonicPolicy::parse(t[1])
                        .ok_or_else(|| format!("line {}: ionic takes none, cores or all, got '{}'", i + 1, t[1]))?;
                }
                "nucleus" if matches!(t.len(), 5 | 6 | 8 | 9) => {
                    let z = t[1]
                        .parse::<u8>()
                        .map_err(|_| format!("line {}: bad Z", i + 1))?;
                    // 6 or 9 fields: the second number is the mass number
                    let (a, rest) = if t.len() == 6 || t.len() == 9 {
                        let a = t[2]
                            .parse::<u16>()
                            .map_err(|_| format!("line {}: bad A '{}'", i + 1, t[2]))?;
                        (canonical_a(z, a), &t[3..])
                    } else {
                        (None, &t[2..])
                    };
                    let pos = [num(rest[0])?, num(rest[1])?, num(rest[2])?];
                    let vel = if rest.len() == 6 {
                        [num(rest[3])?, num(rest[4])?, num(rest[5])?]
                    } else {
                        [0.0; 3]
                    };
                    nuclei.push(SceneNucleus { z, a, pos, vel });
                }
                "nucleus" => {
                    return Err(format!(
                        "line {}: 'nucleus' takes Z x y z [vx vy vz] or Z A x y z [vx vy vz] (5, 6, 8 or 9 fields), got {}",
                        i + 1,
                        t.len()
                    ))
                }
                other => {
                    return Err(format!(
                        "line {}: unknown or malformed directive '{other}'",
                        i + 1
                    ))
                }
            }
        }
        let (grid_n, grid_h) = grid.ok_or("missing 'grid n h' line")?;
        Ok(Scene {
            nuclei,
            charge,
            multiplicity: mult,
            grid_n,
            grid_h,
            sigma,
            ionic,
        })
    }

    /// The viewer's limits for a scene it will hand to the solver: odd `grid_n` in 17..=201,
    /// `grid_h` in 0.08..=0.5 Bohr (the quality levels down to Precise and Custom 0.08 keep the
    /// box: 121³–177³, `crate::quality`), Z in 1..=10 (the element picker's range), finite coordinates
    /// and σ. Physical admissibility (charge, multiplicity, coincident nuclei) is the worker's job.
    pub fn validate(&self) -> Result<(), String> {
        validate_grid(self.grid_n, self.grid_h)?;
        if let Some(s) = self.sigma {
            if !(s.is_finite() && s > 0.0) {
                return Err(format!("sigma {s} must be a positive number"));
            }
        }
        for (i, n) in self.nuclei.iter().enumerate() {
            if !(1..=10).contains(&n.z) {
                return Err(format!("nucleus {i}: Z = {} outside 1..=10", n.z));
            }
            if let Some(a) = n.a {
                if units::isotope_mass_amu_a(n.z, a).is_none() {
                    return Err(format!("nucleus {i}: {}", units::isotopes::unlisted_isotope_message(n.z, a)));
                }
            }
            if n.pos.iter().chain(&n.vel).any(|x| !x.is_finite()) {
                return Err(format!("nucleus {i}: non-finite coordinate"));
            }
        }
        Ok(())
    }

    /// The dynamics driver's scene: masses from (Z, A) (`sim::Nucleus::isotope`, the isotope
    /// table — the only per-element datum besides Z; `a: None` is the most abundant isotope),
    /// velocities as given (Bohr per atomic time unit), the cell from the grid, no external
    /// field (the field is a run setting, not part of the file). Fails for a Z without an
    /// isotope mass (outside 1..=10) or an unlisted A.
    pub fn to_sim(&self) -> Result<sim::Scene, String> {
        let nuclei = self
            .nuclei
            .iter()
            .enumerate()
            .map(|(i, n)| match n.a.or_else(|| units::most_abundant_a(n.z)) {
                Some(a) => sim::Nucleus::isotope(n.z, a, n.pos).map(|nu| nu.with_vel(n.vel)).map_err(|e| format!("nucleus {i}: {e}")),
                None => Err(format!("nucleus {i}: no isotope mass for Z = {}", n.z)),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(sim::Scene {
            nuclei,
            cell: sim::Cell {
                n: self.grid_n,
                h: self.grid_h,
            },
            charge: self.charge,
            multiplicity: self.multiplicity,
            sigma: self.sigma,
            external_field: [0.0; 3],
            ionic: self.ionic,
        })
    }

    /// The file-format view of a dynamics scene (positions and velocities as they are; masses
    /// and the field are dropped — masses follow from (Z, A) on the way back; A is kept only
    /// when it is not the most abundant isotope).
    pub fn from_sim(s: &sim::Scene) -> Scene {
        Scene {
            nuclei: s
                .nuclei
                .iter()
                .map(|n| SceneNucleus {
                    z: n.z,
                    a: canonical_a(n.z, n.a),
                    pos: n.pos,
                    vel: n.vel,
                })
                .collect(),
            charge: s.charge,
            multiplicity: s.multiplicity,
            grid_n: s.cell.n,
            grid_h: s.cell.h,
            sigma: s.sigma,
            ionic: s.ionic,
        }
    }

    pub fn to_text(&self) -> String {
        let mut s = format!(
            "# atomic sandbox scene v1\ngrid {} {}\ncharge {}\n",
            self.grid_n, self.grid_h, self.charge
        );
        if let Some(m) = self.multiplicity {
            s += &format!("multiplicity {m}\n");
        }
        if let Some(x) = self.sigma {
            s += &format!("sigma {x}\n");
        }
        if self.ionic != sim::IonicPolicy::None {
            s += &format!("ionic {}\n", self.ionic.word());
        }
        for n in &self.nuclei {
            s += &format!("nucleus {}", n.z);
            if let Some(a) = n.a.and_then(|a| canonical_a(n.z, a)) {
                s += &format!(" {a}");
            }
            s += &format!(" {} {} {}", n.pos[0], n.pos[1], n.pos[2]);
            if n.vel != [0.0; 3] {
                s += &format!(" {} {} {}", n.vel[0], n.vel[1], n.vel[2]);
            }
            s.push('\n');
        }
        s
    }
}

/// The `h2` preset as it was until 2026-10-14 — R = 1.6 Bohr on 49³ × 0.25 — the geometry the
/// worker and viewer tests were written and pinned for (fix round 1 moved the preset itself to
/// R = 1.45 Bohr, this solver's σ → 0 bond length).
#[cfg(test)]
pub fn h2_r16() -> Scene {
    let mut s = Scene::preset("h2").expect("h2 preset");
    s.nuclei[0].pos = [0.8, 0.0, 0.0];
    s.nuclei[1].pos = [-0.8, 0.0, 0.0];
    s
}

/// The estimated solver's model sidecar for a scene file: next to it, the same name with the
/// extension `.model` (`my.scene` → `my.model`; a name without an extension gains one). The
/// sidecar format is `surrogate::sidecar`'s; the scene format itself is unchanged.
pub fn model_path(scene_path: &str) -> std::path::PathBuf {
    std::path::Path::new(scene_path).with_extension("model")
}

/// The largest grid edge: Custom h = 0.08 on h2o's 14 Bohr box is 177³ (≈ 5.5 M points).
pub const MAX_GRID_N: usize = 201;

/// Grid limits shared by [`Scene::validate`] and the panel's "Apply grid".
pub fn validate_grid(n: usize, h: f64) -> Result<(), String> {
    if n.is_multiple_of(2) || !(17..=MAX_GRID_N).contains(&n) {
        return Err(format!("grid n = {n}: must be odd and in 17..={MAX_GRID_N}"));
    }
    if !(crate::quality::H_MIN..=0.5).contains(&h) {
        return Err(format!("grid h = {h}: must be in {}..=0.5 Bohr", crate::quality::H_MIN));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_accepts_presets_and_rejects_out_of_range_scenes() {
        for name in Scene::PRESETS {
            Scene::preset(name).unwrap().validate().unwrap();
        }
        let ok = Scene::preset("h2").unwrap();
        let with = |f: &dyn Fn(&mut Scene)| {
            let mut s = ok.clone();
            f(&mut s);
            s.validate()
        };
        assert!(with(&|s| s.grid_n = 48).unwrap_err().contains("odd"));
        assert!(with(&|s| s.grid_n = 203).is_err());
        assert!(with(&|s| s.grid_n = 121).is_ok(), "Precise keeps the 12 Bohr box at 121³");
        assert!(with(&|s| s.grid_n = 15).is_err());
        assert!(with(&|s| s.grid_h = 0.05).unwrap_err().contains("h ="));
        assert!(with(&|s| s.grid_h = f64::NAN).is_err());
        assert!(with(&|s| s.nuclei[0].z = 11)
            .unwrap_err()
            .contains("Z = 11"));
        assert!(with(&|s| s.nuclei[1].pos[2] = f64::INFINITY)
            .unwrap_err()
            .contains("non-finite"));
        assert!(with(&|s| s.sigma = Some(0.0)).is_err());
    }

    #[test]
    fn parses_the_documented_example() {
        let s = Scene::parse("# c\ngrid 49 0.25\ncharge 0\nmultiplicity 2\nsigma 0.3\nnucleus 1 0.7 0 0\nnucleus 1 -0.7 0 0 0.1 0 0\n").unwrap();
        assert_eq!(s.grid_n, 49);
        assert_eq!(s.grid_h, 0.25);
        assert_eq!(s.charge, 0);
        assert_eq!(s.multiplicity, Some(2));
        assert_eq!(s.sigma, Some(0.3));
        assert_eq!(s.nuclei.len(), 2);
        assert_eq!(s.nuclei[1].vel, [0.1, 0.0, 0.0]);
    }

    #[test]
    fn round_trips() {
        for name in Scene::PRESETS {
            let s = Scene::preset(name).unwrap();
            assert_eq!(Scene::parse(&s.to_text()).unwrap(), s, "preset {name}");
        }
    }

    /// app ↔ sim: positions and velocities survive both ways bit for bit, masses come from Z
    /// (¹H 1837.15 mₑ), and velocities also survive the text format (written only when non-zero,
    /// so a v1 file without them still parses).
    #[test]
    fn converts_to_and_from_the_dynamics_scene_with_velocities() {
        let mut s = Scene::parse(
            "grid 33 0.3\ncharge 1\nmultiplicity 2\nsigma 0.2\nnucleus 2 0 0 0\nnucleus 1 1.46 0 0 0.0012 -3e-4 0\n",
        )
        .unwrap();
        let d = s.to_sim().unwrap();
        assert_eq!(d.cell, sim::Cell { n: 33, h: 0.3 });
        assert_eq!((d.charge, d.multiplicity, d.sigma), (1, Some(2), Some(0.2)));
        assert_eq!(d.external_field, [0.0; 3]);
        assert_eq!(d.nuclei[1].vel, [0.0012, -3e-4, 0.0]);
        assert_eq!(d.nuclei[0].vel, [0.0; 3]);
        assert!((d.nuclei[1].mass - 1837.15).abs() < 0.05, "m(¹H) = {}", d.nuclei[1].mass);
        assert!(d.nuclei[0].mass > 3.9 * d.nuclei[1].mass);
        assert_eq!(Scene::from_sim(&d), s);
        assert_eq!(Scene::parse(&s.to_text()).unwrap(), s);
        assert!(!s.to_text().contains("nucleus 2 0 0 0 0"), "zero velocities are not written");
        s.nuclei[0].z = 11;
        assert!(s.to_sim().unwrap_err().contains("Z = 11"));
    }

    /// Spec §5.1: `nucleus Z A x y z [vx vy vz]` (6 or 9 fields) carries a mass number, the old
    /// 5/8-field form reads as the most abundant isotope (`a: None`), 7 fields is an error naming
    /// both shapes; the writer emits A only when it is not the most abundant one, so every
    /// preset round-trips byte-identically and a ²H is written and read back.
    #[test]
    fn scene_round_trips_mass_numbers_and_keeps_old_files_byte_identical() {
        let one = |line: &str| Scene::parse(&format!("grid 49 0.25\n{line}\n")).map(|s| s.nuclei[0].clone());
        let d = one("nucleus 1 2 0.8 0 0").unwrap();
        assert_eq!((d.z, d.a, d.pos, d.vel), (1, Some(2), [0.8, 0.0, 0.0], [0.0; 3]));
        assert_eq!(one("nucleus 1 0.8 0 0").unwrap().a, None);
        let dv = one("nucleus 1 2 0.8 0 0 0.1 0 0").unwrap();
        assert_eq!((dv.a, dv.vel), (Some(2), [0.1, 0.0, 0.0]));
        assert_eq!(one("nucleus 1 1 0.8 0 0").unwrap().a, None, "an explicit most-abundant A is the default");
        assert_eq!(one("nucleus 8 18 0 0 0").unwrap().mass_number(), Some(18));
        assert_eq!(one("nucleus 8 0 0 0").unwrap().mass_number(), Some(16));
        let e = one("nucleus 1 0.8 0 0 0.1 0").unwrap_err();
        assert!(e.contains("line 2") && e.contains("Z x y z [vx vy vz]") && e.contains("Z A x y z [vx vy vz]"), "{e}");
        assert!(one("nucleus 1 2.5 0 0 0").unwrap_err().contains("bad A '2.5'"), "a 6-field line with a non-integer A");
        // every preset: parse → write → parse equal, and the text is unchanged by the A field
        for name in Scene::PRESETS {
            let s = Scene::preset(name).unwrap();
            assert!(s.nuclei.iter().all(|n| n.a.is_none()), "preset {name} is written without A");
            assert_eq!(Scene::parse(&s.to_text()).unwrap(), s, "preset {name}");
            assert!(!s.to_text().lines().any(|l| l.split_whitespace().count() == 6 || l.split_whitespace().count() == 9));
        }
        // the shipped preset files themselves are byte-identical through parse → write → parse
        for (name, text) in [("h2", include_str!("../presets/h2.scene")), ("h2o", include_str!("../presets/h2o.scene"))] {
            let s = Scene::parse(text).unwrap();
            assert_eq!(Scene::parse(&s.to_text()).unwrap(), s, "preset {name}");
        }
        // a scene with a ²H writes "nucleus 1 2 …" and reads back with a == Some(2)
        let mut s = Scene::preset("h2").unwrap();
        s.nuclei[0].a = Some(2);
        let text = s.to_text();
        let line = text.lines().find(|l| l.starts_with("nucleus 1 2 ")).expect("the ²H line");
        assert_eq!(line.split_whitespace().count(), 6, "{line}");
        let back = Scene::parse(&text).unwrap();
        assert_eq!(back, s);
        assert_eq!(back.nuclei[0].a, Some(2));
        assert_eq!(back.nuclei[1].a, None);
        // an explicit most-abundant A is dropped on write
        s.nuclei[1].a = Some(1);
        assert!(!s.to_text().contains("nucleus 1 1 "));
        // validate refuses an unlisted A naming the listed ones; to_sim takes the table mass
        s.nuclei[1].a = Some(5);
        assert_eq!(s.validate().unwrap_err(), "nucleus 1: no mass for ⁵H; listed: ¹H ²H ³H");
        assert!(s.to_sim().unwrap_err().contains("⁵H"));
        s.nuclei[1].a = None;
        let sim = s.to_sim().unwrap();
        assert_eq!((sim.nuclei[0].a, sim.nuclei[1].a), (2, 1));
        assert!((sim.nuclei[0].mass - units::amu_to_au_mass(2.014101777844)).abs() < 1e-6);
        assert_eq!(Scene::from_sim(&sim), s, "from_sim gives None for the most abundant isotope and Some for the rest");
    }

    /// The `empty` preset: the default cell (49³ × 0.25 Bohr), charge 0, multiplicity automatic,
    /// no nuclei — a file with no `nucleus` line parses, round-trips through the text format and
    /// through the dynamics scene, and validates.
    #[test]
    fn the_empty_preset_is_a_cell_without_nuclei_and_round_trips() {
        assert_eq!(Scene::PRESETS[0], "empty", "empty is listed first");
        let e = Scene::preset("empty").unwrap();
        assert_eq!(
            e,
            Scene { nuclei: Vec::new(), charge: 0, multiplicity: None, grid_n: 49, grid_h: 0.25, sigma: None, ionic: sim::IonicPolicy::None }
        );
        e.validate().unwrap();
        let text = e.to_text();
        assert!(!text.contains("nucleus"), "{text}");
        assert_eq!(Scene::parse(&text).unwrap(), e);
        assert_eq!(Scene::parse("grid 33 0.3\n").unwrap().nuclei.len(), 0, "only the grid line is required");
        let d = e.to_sim().unwrap();
        assert!(d.nuclei.is_empty());
        assert_eq!((d.cell.n, d.cell.h, d.charge, d.multiplicity), (49, 0.25, 0, None));
        assert_eq!(Scene::from_sim(&d), e);
    }

    /// M7 Task 9 (spec §4.7): the `ionic` line round-trips through the text format and the dynamics scene,
    /// a file without it means `None` (an old file keeps the physics it was written for), `None` is not
    /// written, the presets are the files as written (all-electron), and a bad word names its line.
    #[test]
    fn ionic_key_round_trips_and_defaults_to_none() {
        for p in sim::IonicPolicy::ALL {
            let mut s = Scene::preset("h2o").unwrap();
            s.ionic = p;
            let text = s.to_text();
            assert_eq!(text.contains("ionic "), p != sim::IonicPolicy::None, "{text}");
            let back = Scene::parse(&text).unwrap();
            assert_eq!(back, s);
            assert_eq!(back.ionic, p);
            assert_eq!(s.to_sim().unwrap().ionic, p);
            assert_eq!(Scene::from_sim(&s.to_sim().unwrap()), s);
        }
        assert_eq!(Scene::parse("grid 33 0.3\nnucleus 1 0 0 0\n").unwrap().ionic, sim::IonicPolicy::None);
        for name in Scene::PRESETS {
            assert_eq!(Scene::preset(name).unwrap().ionic, sim::IonicPolicy::None, "{name}: the presets are the files as written");
        }
        let e = Scene::parse("grid 33 0.3\nionic everything\n").unwrap_err();
        assert!(e.contains("line 2") && e.contains("none, cores or all"), "{e}");
    }

    #[test]
    fn errors_name_the_line() {
        let e = Scene::parse("grid 49 0.25\nnucleus 1 0 0\n").unwrap_err();
        assert!(e.contains("line 2"), "{e}");
        assert!(Scene::parse("nucleus 1 0 0 0\n")
            .unwrap_err()
            .contains("grid"));
        assert!(Scene::parse("grid 49 0.25\nbogus 1\n")
            .unwrap_err()
            .contains("bogus"));
    }

    #[test]
    fn presets_are_neutral_and_sane() {
        let w = Scene::preset("h2o").unwrap();
        assert_eq!(w.nuclei.iter().map(|n| n.z as u32).sum::<u32>(), 10);
        let d = ((w.nuclei[1].pos[0] - w.nuclei[0].pos[0]).powi(2)
            + (w.nuclei[1].pos[1] - w.nuclei[0].pos[1]).powi(2))
        .sqrt();
        assert!((d - 1.81).abs() < 0.02, "O–H = {d} Bohr");
    }

    #[test]
    fn the_model_sidecar_sits_next_to_the_scene_file() {
        use std::path::PathBuf;
        assert_eq!(model_path("my.scene"), PathBuf::from("my.model"));
        assert_eq!(model_path("runs/h2.scene"), PathBuf::from("runs/h2.model"));
        assert_eq!(model_path("plain"), PathBuf::from("plain.model"));
    }
}
