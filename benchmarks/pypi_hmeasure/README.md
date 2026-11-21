# pyhmeasure against hmeasure on PyPI

[hmeasure](https://pypi.org/project/hmeasure/) on PyPI (version 0.1.6, by Lyubomir Danov) is a Python
translation of the R [hmeasure](https://cran.r-project.org/package=hmeasure) package. This benchmark compares
it with pyhmeasure 0.1.0, which wraps the Rust [hmeasure](https://crates.io/crates/hmeasure) crate.

## They compute the same H-measure

The two packages agree to within 1e-12 on every case the tests check: 120 random cases, with balanced and
imbalanced classes, tied and untied scores, and classes as small as one observation, plus the benchmark data
at every size. In practice the largest difference seen was about 2e-14.

Their cost distributions have to be matched for this. hmeasure on PyPI follows Hand and Anagnostopoulos
(2014) and uses a Beta(2, 1 + 1/SR) cost distribution, where the severity ratio SR defaults to n1/n0, the
ratio of the class sizes. So:

| hmeasure on PyPI                     | pyhmeasure                                    |
|--------------------------------------|-----------------------------------------------|
| `h_score(y, scores, severity_ratio=1.0)` | `PyHmeasure(2.0, 2.0, class0, class1)`    |
| `h_score(y, scores)` (the default)   | `PyHmeasure(2.0, 1.0 + n0 / n1, class0, class1)` |

Both use the same algorithm in outline: the convex hull of the ROC curve, then the cost integrals in closed
form through the Beta CDF. Both scale as O(n log n), dominated by sorting the scores.

## Speed

Median times, with a Beta(2, 2) cost distribution. The scores come from a classifier that separates the
classes fairly well, with five class 0 observations for every four in class 1, and no tied scores.

| Scores    | hmeasure on PyPI | pyhmeasure | pyhmeasure is faster by |
|-----------|-----------------:|-----------:|------------------------:|
| 3,600     |          5.04 ms |    1.38 ms |                    3.7x |
| 36,000    |          39.8 ms |    11.8 ms |                    3.4x |
| 360,000   |           457 ms |     140 ms |                    3.3x |
| 3,600,000 |           5.30 s |     1.43 s |                    3.7x |

Measured on an Intel Core i9-9880H (2.3 GHz) under macOS 26.6, with Python 3.13, NumPy 2.5, SciPy 1.18 and
scikit-learn 1.9. Absolute times depend on the machine; the ratios are the point of the comparison.

Both packages are timed from the same inputs, a NumPy array of 0/1 labels and a NumPy array of scores, which
is the form hmeasure on PyPI takes. pyhmeasure takes the scores of each class separately, so its times include
splitting the scores by label. That is why they are higher than the 0.76 ms for 3,600 scores in the
[main README](../../README.md), which passes in scores already split into lists.

Where the time goes in hmeasure on PyPI, from a profile at 360,000 scores: about 80% is in building the convex
hull of the ROC curve, which assembles the ROC points with a Python-level `list(zip(...))` and then runs
SciPy's general-purpose two-dimensional convex hull (Qhull). Most of the rest is scikit-learn's `roc_curve`.
pyhmeasure builds the hull in a single pass over the ROC points, which are already in order.

## Other differences

- **Scores outside [0, 1].** hmeasure on PyPI requires the scores to lie between the smallest and largest
  label, so with 0/1 labels a classifier's raw outputs, such as logits, are rejected with a `TypeError`.
  pyhmeasure accepts any real-valued scores; only their ordering matters.
- **Invalid inputs.** A NaN score gives a `TypeError` from hmeasure on PyPI saying the scores must not be larger
  than the largest label; pyhmeasure raises a `ValueError` saying which class contains a NaN score. With only
  one class present, both raise an error.
- **Output.** hmeasure on PyPI returns the H-measure. pyhmeasure also keeps the ROC curve and its convex hull
  in the Rust results, though the Python wrapper currently exposes only the H-measure.
- **Dependencies.** hmeasure on PyPI needs NumPy, SciPy and scikit-learn. pyhmeasure has no Python
  dependencies, but is a compiled package, so it needs to be installed pre-built or built from source with
  Rust.

## Running it

The environment takes pyhmeasure from conda-forge. hmeasure on PyPI is not on conda-forge, so the environment
installs it with pip:

```
conda env create -f environment.yml
conda activate hmeasure-benchmark
pytest
```

from this directory. The run takes about a minute. `pytest --benchmark-disable` runs only the agreement
checks, in about 20 seconds.
