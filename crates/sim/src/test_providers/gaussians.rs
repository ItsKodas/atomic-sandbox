//! Analytic Gaussian densities for the bond-detector and reference tests (M6 Task 3): a
//! [`ForceProvider`] with zero forces whose `sample_density` is exact ([`Gaussians`]), and one
//! whose density is evaluated on a cell's nodes and sampled trilinearly ([`GriddedGaussians`]) —
//! the operator the Kohn-Sham providers apply — for the same-operator bias tests.
use crate::bonds::{node_position, trilinear_blend};
use crate::provider::{EvalStats, ForceProvider, ForceResult, ProviderError};
use crate::scene::{Cell, Scene};

/// One term amp·exp(−|p − centre|²/2w²).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Term {
    pub centre: [f64; 3],
    pub amp: f64,
    pub w: f64,
}

impl Term {
    pub fn at(&self, p: [f64; 3]) -> f64 {
        let d: [f64; 3] = std::array::from_fn(|a| p[a] - self.centre[a]);
        self.amp * (-(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) / (2.0 * self.w * self.w)).exp()
    }
}

/// Σ of Gaussian terms: `fixed` ones in space, plus — when configured — one per nucleus of the
/// evaluated scene (the same (A, w) for every Z: a stand-in atom) and one bump at the midpoint
/// of every pair closer than `r_pair` (a stand-in bond). Energy and forces are zero (the scene
/// does not move; the driver tests only need ticks to run). `sample_density` answers exactly
/// after an evaluation and `None` after a `reset` (the Kohn-Sham adapters' rule); [`sample`]
/// (Self::sample) evaluates the current terms regardless.
#[derive(Clone, Debug)]
pub struct Gaussians {
    fixed: Vec<Term>,
    per_atom: Option<(f64, f64)>,
    bump: Option<(f64, f64, f64)>,
    current: Option<Vec<Term>>,
}

impl Gaussians {
    /// Terms fixed in space (the analytic fixtures).
    pub fn fixed(terms: Vec<Term>) -> Gaussians {
        Gaussians { fixed: terms, per_atom: None, bump: None, current: None }
    }

    /// One Gaussian (amp, w) on every nucleus of the evaluated scene.
    pub fn per_atom(amp: f64, w: f64) -> Gaussians {
        Gaussians { fixed: Vec::new(), per_atom: Some((amp, w)), bump: None, current: None }
    }

    /// Plus a bump (amp, w) at the midpoint of every pair closer than `r_pair`.
    pub fn with_pair_bump(self, amp: f64, w: f64, r_pair: f64) -> Gaussians {
        Gaussians { bump: Some((amp, w, r_pair)), ..self }
    }

    /// The terms the density of `scene` is made of.
    pub fn terms_for(&self, scene: &Scene) -> Vec<Term> {
        let mut t = self.fixed.clone();
        if let Some((amp, w)) = self.per_atom {
            t.extend(scene.nuclei.iter().map(|n| Term { centre: n.pos, amp, w }));
        }
        if let Some((amp, w, r_pair)) = self.bump {
            let n = &scene.nuclei;
            for i in 0..n.len() {
                for j in i + 1..n.len() {
                    let d: [f64; 3] = std::array::from_fn(|a| n[j].pos[a] - n[i].pos[a]);
                    if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() < r_pair {
                        t.push(Term { centre: std::array::from_fn(|a| 0.5 * (n[i].pos[a] + n[j].pos[a])), amp, w });
                    }
                }
            }
        }
        t
    }

    /// The current terms (the last evaluation's, else the fixed ones).
    pub fn terms(&self) -> &[Term] {
        self.current.as_deref().unwrap_or(&self.fixed)
    }

    pub fn density(&self, p: [f64; 3]) -> f64 {
        self.terms().iter().map(|t| t.at(p)).sum()
    }

    pub fn sample(&self, points: &[[f64; 3]]) -> Vec<f64> {
        points.iter().map(|&p| self.density(p)).collect()
    }
}

impl ForceProvider for Gaussians {
    fn evaluate(&mut self, scene: &Scene, _a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        self.current = Some(self.terms_for(scene));
        Ok(ForceResult { energy: 0.0, forces: vec![[0.0; 3]; scene.nuclei.len()], stats: EvalStats::default(), display: None })
    }

    fn name(&self) -> &'static str {
        "gaussians"
    }

    fn reset(&mut self) {
        self.current = None;
    }

    fn sample_density(&mut self, points: &[[f64; 3]]) -> Option<Vec<f64>> {
        self.current.as_ref().map(|t| points.iter().map(|&p| t.iter().map(|x| x.at(p)).sum()).collect())
    }

    /// A clone with no current terms (the references' own instance).
    fn side_solver(&self) -> Option<Box<dyn ForceProvider>> {
        Some(Box::new(Gaussians { current: None, ..self.clone() }))
    }
}

/// Fixed terms evaluated on the nodes of `cell` and sampled by trilinear interpolation
/// ([`trilinear_blend`]: `qm::Grid::trilinear`'s operator), 0 outside the grid. `evaluate`
/// ignores the scene (the density is the fixed terms'); `sample_density` answers after an
/// evaluation and `None` after a reset.
#[derive(Clone, Debug)]
pub struct GriddedGaussians {
    cell: Cell,
    values: Vec<f64>,
    current: bool,
}

impl GriddedGaussians {
    pub fn new(terms: Vec<Term>, cell: Cell) -> GriddedGaussians {
        let n = cell.n;
        let mut values = Vec::with_capacity(n * n * n);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let p = node_position(&cell, [i, j, k]);
                    values.push(terms.iter().map(|t| t.at(p)).sum());
                }
            }
        }
        GriddedGaussians { cell, values, current: false }
    }

    /// The node value at (i, j, k).
    fn node(&self, ijk: [usize; 3]) -> f64 {
        let n = self.cell.n;
        self.values[ijk[0] + n * (ijk[1] + n * ijk[2])]
    }

    pub fn sample(&self, points: &[[f64; 3]]) -> Vec<f64> {
        points.iter().map(|&p| trilinear_blend(&self.cell, p, |ijk| self.node(ijk))).collect()
    }
}

impl ForceProvider for GriddedGaussians {
    fn evaluate(&mut self, scene: &Scene, _a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        self.current = true;
        Ok(ForceResult { energy: 0.0, forces: vec![[0.0; 3]; scene.nuclei.len()], stats: EvalStats::default(), display: None })
    }

    fn name(&self) -> &'static str {
        "gridded gaussians"
    }

    fn reset(&mut self) {
        self.current = false;
    }

    fn sample_density(&mut self, points: &[[f64; 3]]) -> Option<Vec<f64>> {
        self.current.then(|| self.sample(points))
    }

    fn side_solver(&self) -> Option<Box<dyn ForceProvider>> {
        Some(Box::new(GriddedGaussians { current: false, ..self.clone() }))
    }
}
