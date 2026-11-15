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
    hm = pyhmeasure.PyHmeasure(cost_distribution_alpha, cost_distribution_beta, c0scores, c1scores)
    print(f"H-Measure:{hm.h}")
```

Class 0 is the class expected to receive the lower scores. Invalid inputs (cost distribution parameters that
are not greater than zero, a class with no scores, or a NaN score) raise a `ValueError`.

The scores can be passed in as lists or numpy arrays, as above, or loaded from a csv file by the Rust code by
passing the path of the file instead: `pyhmeasure.PyHmeasure(alpha, beta, None, None, path)`. The examples folder
and tests illustrate both.

**BENCHMARKS**

The tests include benchmarks comparing this Rust implementation with a pure python implementation of the same
algorithm (in `python_src`): both build the convex hull of the ROC curve in a single pass and integrate a Beta
cost distribution in closed form. Illustrative results, using version 0.1.0 of the hmeasure crate, a release
build, Python 3.13 and a Beta(2, 2) cost distribution:

| 3,600 scores (2,000 + 1,600)            | Median time | Relative |
|-----------------------------------------|------------:|---------:|
| Rust, scores passed in                  |     0.76 ms |      1.0 |
| Rust, scores loaded from csv            |     1.48 ms |      1.9 |
| Python, scores passed in                |     5.88 ms |      7.8 |
| Python, scores loaded from csv          |     6.70 ms |      8.8 |

| 360,000 scores (200,000 + 160,000)      | Median time | Relative |
|-----------------------------------------|------------:|---------:|
| Rust, scores passed in                  |       92 ms |      1.0 |
| Rust, scores loaded from csv            |      159 ms |      1.7 |
| Python, scores passed in                |      707 ms |      7.7 |
| Python, scores loaded from csv          |      724 ms |      7.8 |

On the 3,600 score benchmark with the scores passed in, version 0.1.0 of the hmeasure crate is about 25 times
faster than version 0.0.5, which the previous version of this wrapper used (0.76 ms against 19.8 ms). Absolute timings depend on the machine; the ratios are the
point of the comparison.

To run the tests and benchmarks, install the python packages they use into the virtual environment described
below and run pytest from the root of the project:

`pip install numpy pandas scipy pytest pytest-benchmark`\
`pytest tests`

**INSTALLATION**

The Rust implementation is exposed to Python using the pyo3 crate, which requires Python 3.9 or later, Rust 1.85 or later
and maturin 1.9.4 or later. Within a newly created and activated python virtual environment,
install maturin (https://github.com/PyO3/maturin) and run `maturin develop --release` to build this rust crate and install it as a 
python module:

`cd your_path/to/project/pyhmeasure` : navigate to this project (after you have git cloned it to your machine)\
`python3 -m venv .env` : create a new virtual environment\
`source .env/bin/activate` : activate the venv\
`pip install maturin` : install the pyo3 maturin package in the venv\
`maturin develop --release` : run maturin develop on this project to create the pyhmeasure python package 
and install it in this active venv

The `--release` flag builds the Rust code with optimisations; without it the H-Measure calculation is several times slower.
If maturin reports that both `VIRTUAL_ENV` and `CONDA_PREFIX` are set, run `unset CONDA_PREFIX` first.

**FURTHER READING**

A discussion of H-measure with further examples is provided in Chapter 2 of:›
<https://github.com/robinwiseman/finML/blob/aa12845f01454c24f36f4df0d1cb6e0993ea7c7f/src/finML_2022.pdf>

Hand, D.J., Anagnostopoulos, C. \
Notes on the H-measure of classifier performance. \
Adv Data Anal Classif 17, 109–124 (2023). \
<https://doi.org/10.1007/s11634-021-00490-3>