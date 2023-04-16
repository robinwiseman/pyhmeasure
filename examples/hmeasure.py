import os
import pandas as pd

# A python implementation (for comparison)
from python_src.hmeasure.datagen import DataGenBinaryClassifierScores
from python_src.hmeasure.h_measure import beta, CostRatioDensity, HMeasure

# the Rust implementation exposed in python
import pyhmeasure

# Illustrate calling the Rust implementation of H-Measure by passing in the scores directly.
# This is slower than loading the scores from file natively in Rust (see further example below).
c0scores = [0.0, 0.01, 0.02, 0.03, 0.2, 0.6, 0.66, 0.7]
c1scores = [0.3, 0.35, 0.36, 0.42, 0.5, 0.8, 0.82, 0.99]
cd_alpha = 2.0
cd_beta = 2.0
hm = pyhmeasure.PyHmeasure(cd_alpha, cd_beta, c0scores, c1scores, None)
print(f"H-Measure:{hm.h}")

# Generate dummy score data and write it to csv, before then running H-Measure, loading the scores from
# file.
class_params = {
            'class0_alpha': 2.0,
            'class0_beta': 6.0,
            'class1_alpha': 6.0,
            'class1_beta': 2.0
        }
c0_sample_size = 2000
c1_sample_size = 1800  # slightly imbalanced classification problem
dg_bcs = DataGenBinaryClassifierScores(class_params=class_params,
                                       c0_sample_size=c0_sample_size,
                                       c1_sample_size=c1_sample_size)
score_samples = dg_bcs.generate_samples()
c0scores = score_samples.get('class_0')
c1scores = score_samples.get('class_1')
df = pd.DataFrame({"score_0": pd.Series(c0scores), "score_1": pd.Series(c1scores)})
path = os.path.join(os.getcwd(),"scores.csv")
df.to_csv(path);

cd_alpha = 2.0
cd_beta = 2.0
hm = pyhmeasure.PyHmeasure(cd_alpha, cd_beta, None, None, path)
print(f"H-Measure (rust implementation):{hm.h}")

crd = CostRatioDensity(beta(2.0, 2.0))
hm = HMeasure(cost_distribution=crd)
_, _, h = hm.h_measure(score_samples)
print(f"H-Measure (python implementation):{h}")
