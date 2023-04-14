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
A discussion with further examples (and reference to a pure python implementation) is provided in Chapter 2 of:
<https://github.com/robinwiseman/finML/blob/aa12845f01454c24f36f4df0d1cb6e0993ea7c7f/src/finML_2022.pdf>
