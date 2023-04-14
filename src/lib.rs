/*!
pyhmeasure.PyHmeasure enables running the rust crate HMeasure::h_measure from Python.

<https://crates.io/crates/hmeasure>

**H-Measure**

The hmeasure crate provides a Rust implementation of the H-measure. The **H-measure is a coherent alternative**
to the widely used AUC measure (Area Under the ROC Curve) for assessing the relative quality
of binary classifiers. Whereas the AUC implicitly assumes a different cost distribution for the
"cost of being wrong" depending on the classifier it is applied to, the H-Measure, by contrast, enables
the researcher to fix the cost distribution consistently across all classifiers studied. The "cost of
being wrong" is an inherent property of the subject being modelled and should not depend on the
specific model being used to model the subject. H-Measure enables that consistency, whereas AUC does not.

The H-measure was introduced by David J. Hand in the paper:

"Measuring classifier performance: a coherent alternative to the area under the ROC curve"
Mach Learn (2009) 77: 103–123
<https://link.springer.com/article/10.1007/s10994-009-5119-5>
*/
use pyo3::prelude::*;
use pyo3::types::PyList;

use hmeasure::{CostRatioDensity, HMeasure};
use hmeasure::{BetaParams, BinaryClassScores};

#[pyclass]
pub struct PyHmeasure {
    pub h: f64,
    pub convex_hull: Vec<Vec<f64>>,
    pub roc_curve: Vec<Vec<f64>>,
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
        self.vec_to_list("class0_score")
    }
    #[getter]
    fn class1_score(&self) -> PyResult<Py<PyList>> {
        self.vec_to_list("class1_score")
    }
    #[getter]
    fn convex_hull(&self) -> PyResult<Py<PyList>> {
        self.vecvec_to_list("convex_hull")
    }
    #[getter]
    fn roc_curve(&self) -> PyResult<Py<PyList>> {
        self.vecvec_to_list("roc_curve")
    }
    fn vec_to_list(&self, member: &str) -> PyResult<Py<PyList>> {
        let missing= vec![0.0];
        let vec = match member {
            "class0_score" => &self.class0_score,
            "class1_score" => &self.class1_score,
            _ => &missing
        };
        Python::with_gil(|py| {
            let list = PyList::empty(py);
            for item in vec {
                list.append(item)?;
            }
            Ok(list.into())
        })
    }
    fn vecvec_to_list(&self, member: &str) -> PyResult<Py<PyList>> {
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
                    innerlist.append(item)?;
                }
                list.append(innerlist)?;
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

