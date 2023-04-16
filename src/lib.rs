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
use csv::Reader;
use std::io::BufReader;
use std::fs::File;

use hmeasure::{CostRatioDensity, HMeasure};
use hmeasure::{BetaParams, BinaryClassScores};

fn load_scores(file_path: &str) -> (Vec<f64>,Vec<f64>){
    let file = File::open(file_path).unwrap();
    let buf_reader = BufReader::new(file);
    let mut reader = Reader::from_reader(buf_reader);

    let mut score0 = vec![];
    let mut score1 = vec![];

    for result in reader.records() {
        let record = result.unwrap();
        let val1 = record.get(1).unwrap();
        if val1.len() > 0 {
            let val1_num = val1.parse::<f64>().unwrap();
            score0.push(val1_num);
        }

        let val2 = record.get(2).unwrap();
        if val2.len() > 0 {
            let val2_num = val2.parse::<f64>().unwrap();
            score1.push(val2_num);
        }
    }

    (score0, score1)
}

fn get_scores(c0scores: Option<Vec<f64>>, c1scores: Option<Vec<f64>>,
               score_path: Option<&str>) -> BinaryClassScores{

    if c0scores.is_none() || c1scores.is_none() {
            if score_path.is_none() {
                panic!("Either scores must be provided or a path to a csv to retrieve scores")
            } else {
                let Some(path) = score_path else{ panic!("Problem getting the path to scores file")};
                let (c0s, c1s) = load_scores(path);
                BinaryClassScores { class0: c0s, class1: c1s }
            }
    }
    else {
            BinaryClassScores { class0: c0scores.unwrap(), class1: c1scores.unwrap() }
    }
}

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
    pub fn new(cost_density_alpha: f64, cost_density_beta: f64,
               c0scores: Option<Vec<f64>>, c1scores: Option<Vec<f64>>,
               score_path: Option<&str>) -> PyHmeasure {

        let mut bcs = get_scores(c0scores,c1scores,score_path);
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
}

#[pymodule]
fn pyhmeasure(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyHmeasure>()?;
    Ok(())
}

