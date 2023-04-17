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

A simple example is as follows:

 ```python
    import pyhmeasure
    # scores from your classifier model:
    c0scores = [0.0, 0.01, 0.02, 0.03, 0.2, 0.6, 0.66, 0.7]
    c1scores = [0.3, 0.35, 0.36, 0.42, 0.5, 0.8, 0.82, 0.99]
    # specify the parameters of the cost distribution (a Beta distribution):
    cost_distribution_alpha = 2.0
    cost_distribution_beta = 2.0
    hm = pyhmeasure.PyHmeasure(c0scores, c1scores, cost_distribution_alpha, cost_distribution_beta)
    print(f"H-Measure:{hm.h}")
```

In practice, for large score arrays, it is faster to load the scores from file than pass them in as arguments to 
PyHmeasure. The examples folder and tests (with benchmark tests) illustrate how to do this. For score arrays with a length of 
around 2000 scores, illustrative benchmark comparisons look something like:

![alt text](img/benchmark_comparison.png?raw=true)

**INSTALLATION**

The Rust implementation is exposed to Python using the pyo3 crate. Within a newly created and activated python virtual environment,
install maturin (https://github.com/PyO3/maturin) and run `maturin develop` to build this rust crate and install it as a 
python module:

`cd your_path/to/project/pyhmeasure` : navigate to this project (after you have git cloned it to your machine)\
`python3 -m venv .env` : create a new virtual environment\
`source .env/bin/activate` : activate the venv\
`pip install maturin` : install the pyo3 maturin package in the venv\
`maturin develop` : run maturin develop on this project to create the pyhmeasure python package 
and install it in this active venv

**FURTHER READING**

A discussion of H-measure with further examples is provided in Chapter 2 of:›
<https://github.com/robinwiseman/finML/blob/aa12845f01454c24f36f4df0d1cb6e0993ea7c7f/src/finML_2022.pdf>

Hand, D.J., Anagnostopoulos, C. \
Notes on the H-measure of classifier performance. \
Adv Data Anal Classif 17, 109–124 (2023). \
<https://doi.org/10.1007/s11634-021-00490-3>