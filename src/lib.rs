use pyo3::prelude::*;
use pyo3::types::PyList;


use hmeasure::{CostRatioDensity, HMeasureResults, HMeasure};
use hmeasure::{BetaParams, BinaryClassScores};

#[pyclass]
pub struct PyHmeasure {
    pub h: f64,
    pub convex_hull: Vec<Vec<f64>>,
    pub roc_curve: Vec<Vec<f64>>,
    pub int_components: Vec<Vec<f64>>,
    pub class0_score: Vec<f64>,
    pub class1_score: Vec<f64>
}

#[pymethods]
impl PyHmeasure {
    #[new]
    pub fn new(c0scores: Vec<f64>, c1scores: Vec<f64>,
               cost_density_alpha: f64, cost_density_beta: f64) -> PyHmeasure {
        let mut bcs = BinaryClassScores { class0: c0scores,
                                          class1: c1scores };
        let beta_params = BetaParams { alpha: cost_density_alpha,
                                       beta: cost_density_beta };
        let crd = CostRatioDensity::new(beta_params);
        // calculate the H-Measure given the cost ratio density and scores
        let mut hm = HMeasure::new(crd, None, None);
        let hmr = hm.h_measure(&mut bcs);
        PyHmeasure {
                    h: hmr.h,
                    convex_hull: hmr.convex_hull,
                    roc_curve: hmr.roc_curve,
                    int_components: hmr.int_components,
                    class0_score: hmr.class0_score,
                    class1_score: hmr.class1_score
                }
    }
    #[getter]
    fn h(&self) -> f64 {
        self.h
    }
    #[getter]
    fn class0_score(&self) -> PyResult<Py<PyList>> {
        self.Vec_to_List("class0_score")
    }
    #[getter]
    fn class1_score(&self) -> PyResult<Py<PyList>> {
        self.Vec_to_List("class1_score")
    }
    #[getter]
    fn convex_hull(&self) -> PyResult<Py<PyList>> {
        self.VecVec_to_List("convex_hull")
    }
    #[getter]
    fn roc_curve(&self) -> PyResult<Py<PyList>> {
        self.VecVec_to_List("roc_curve")
    }
    fn Vec_to_List(&self, member: &str) -> PyResult<Py<PyList>> {
        let missing= vec![0.0];
        let vec = match member {
            "class0_score" => &self.class0_score,
            "class1_score" => &self.class1_score,
            _ => &missing
        };
        Python::with_gil(|py| {
            let list = PyList::empty(py);
            for item in vec {
                list.append(item);
            }
            Ok(list.into())
        })
    }
    fn VecVec_to_List(&self, member: &str) -> PyResult<Py<PyList>> {
        let missing= vec![vec![0.0]];
        let vec = match member {
            "convex_hull" => &self.convex_hull,
            "roc_curve" => &self.roc_curve,
            _ => &missing
        };
        Python::with_gil(|py| {
            let list = PyList::empty(py);
            for inner_vec in vec {
                let innerlist = PyList::empty(py);
                for item in inner_vec {
                    innerlist.append(item);
                }
                list.append(innerlist);
            }
            Ok(list.into())
        })
    }
}

#[pymodule]
fn pyhmeasure(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyHmeasure>()?;
    Ok(())
}

