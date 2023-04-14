use pyo3::prelude::*;

use hmeasure::{CostRatioDensity, HMeasureResults, HMeasure};
use hmeasure::{BetaParams, BinaryClassScores};

#[pyclass]
pub struct PyHmeasure {
    pub scores: BinaryClassScores,
    pub cost_density_alpha: f64,
    pub cost_density_beta: f64
}

#[pymethods]
impl PyHmeasure {
    #[new]
    pub fn new(c0scores: Vec<f64>, c1scores: Vec<f64>,
               cost_density_alpha: f64, cost_density_beta: f64) -> PyHmeasure {
        let bcs = BinaryClassScores {
            class0: c0scores,
            class1: c1scores
        };

        PyHmeasure {
            scores: bcs,
            cost_density_alpha: cost_density_alpha,
            cost_density_beta: cost_density_beta
        }
    }
    pub fn h_measure(&mut self) -> f64 {
        let beta_params = BetaParams { alpha: self.cost_density_alpha,
                                       beta: self.cost_density_beta };
        let crd = CostRatioDensity::new(beta_params);
        // calculate the H-Measure given the cost ratio density and scores
        let mut hm = HMeasure::new(crd, None, None);
        let hmr = hm.h_measure(&mut self.scores);
        hmr.h
    }
}

#[pymodule]
fn pyhmeasure(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyHmeasure>()?;
    Ok(())
}

