//! Scene text format v1 (see module tests for the grammar) and embedded presets.
//!
//! One directive per line, `#` starts a comment, fields are whitespace-separated:
//! ```text
//! # atomic sandbox scene v1
//! grid 49 0.25            # n h   (cubic, Bohr)
//! charge 0
//! multiplicity 2          # optional
//! sigma 0.25              # optional; default = h
//! nucleus 1  0.7 0 0      # Z x y z [vx vy vz]
//! nucleus 1 -0.7 0 0
//! ```
//! Positions are in Bohr, velocities in Bohr per atomic time unit.

#[derive(Clone, Debug, PartialEq)]
pub struct SceneNucleus {
    pub z: u8,
    pub pos: [f64; 3],
    pub vel: [f64; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Scene {
    pub nuclei: Vec<SceneNucleus>,
    pub charge: i32,
    pub multiplicity: Option<u32>,
    pub grid_n: usize,
    pub grid_h: f64,
    pub sigma: Option<f64>,
}

impl Scene {
    pub const PRESETS: &'static [&'static str] = &["h", "h2", "he", "lih", "h2o"];

    pub fn preset(name: &str) -> Option<Scene> {
        let text = match name {
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
                "nucleus" if t.len() == 5 || t.len() == 8 => {
                    let z = t[1]
                        .parse::<u8>()
                        .map_err(|_| format!("line {}: bad Z", i + 1))?;
                    let pos = [num(t[2])?, num(t[3])?, num(t[4])?];
                    let vel = if t.len() == 8 {
                        [num(t[5])?, num(t[6])?, num(t[7])?]
                    } else {
                        [0.0; 3]
                    };
                    nuclei.push(SceneNucleus { z, pos, vel });
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
        })
    }

    /// The viewer's limits for a scene it will hand to the solver: odd `grid_n` in 17..=97,
    /// `grid_h` in 0.1..=0.5 Bohr, Z in 1..=10 (the element picker's range), finite coordinates
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
            if n.pos.iter().chain(&n.vel).any(|x| !x.is_finite()) {
                return Err(format!("nucleus {i}: non-finite coordinate"));
            }
        }
        Ok(())
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
        for n in &self.nuclei {
            s += &format!("nucleus {} {} {} {}", n.z, n.pos[0], n.pos[1], n.pos[2]);
            if n.vel != [0.0; 3] {
                s += &format!(" {} {} {}", n.vel[0], n.vel[1], n.vel[2]);
            }
            s.push('\n');
        }
        s
    }
}

/// Grid limits shared by [`Scene::validate`] and the panel's "Apply grid".
pub fn validate_grid(n: usize, h: f64) -> Result<(), String> {
    if n.is_multiple_of(2) || !(17..=97).contains(&n) {
        return Err(format!("grid n = {n}: must be odd and in 17..=97"));
    }
    if !(0.1..=0.5).contains(&h) {
        return Err(format!("grid h = {h}: must be in 0.1..=0.5 Bohr"));
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
        assert!(with(&|s| s.grid_n = 99).is_err());
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
}
