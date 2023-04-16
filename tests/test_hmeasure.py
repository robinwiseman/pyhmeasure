import os
import pytest
import numpy as np
import pandas as pd
from scipy.stats import beta

import pyhmeasure # the python module exposing the Rust implementation of H-Measure

# a local python implementation of H-Measure for comparison in the test + utilities for generating test examples
from python_src.hmeasure.datagen import DataGenBinaryClassifierScores
from python_src.hmeasure.h_measure import HMeasure, CostRatioDensity


class TestHmeasure:
    @pytest.fixture(scope="class")
    def score_path(self):
        # Generate dummy score data for class0 and class1 to be used by the tests
        class_params = {
            'class0_alpha': 2.0,
            'class0_beta': 6.0,
            'class1_alpha': 6.0,
            'class1_beta': 2.0
        }
        c0_sample_size = 2000
        c1_sample_size = 1600  # slightly imbalanced classification problem
        dg_bcs = DataGenBinaryClassifierScores(class_params=class_params,
                                               c0_sample_size=c0_sample_size,
                                               c1_sample_size=c1_sample_size)
        score_samples = dg_bcs.generate_samples()
        c0scores = score_samples.get('class_0')
        c1scores = score_samples.get('class_1')
        df = pd.DataFrame({"score_0": pd.Series(c0scores), "score_1": pd.Series(c1scores)})
        path = os.path.join(os.getcwd(),"scores.csv")
        df.to_csv(path);

        return path

    def test_python_hmeasure(self, score_path, benchmark):
        benchmark(self.run_pure_python, score_path)

    def run_pure_python(self, score_path):
        # call the pure python implementation of H-Measure
        scores = pd.read_csv(score_path)
        scores = {'class_0': scores['score_0'].dropna(),
                  'class_1': scores['score_1'].dropna()}
        crd = CostRatioDensity(beta(2.0, 2.0))
        hm = HMeasure(cost_distribution=crd)
        _ = hm.h_measure(scores)


    def test_rust_hmeasure(self, score_path, benchmark):
        benchmark(self.run_rust, score_path)

    def run_rust(self, score_path):
        # call the rust implementation of H-Measure exposed in python
        cd_alpha = 2.0
        cd_beta = 2.0
        _ = pyhmeasure.PyHmeasure(cd_alpha, cd_beta, None, None, score_path)

    def test_assert_almost_equal(self, score_path):
        cd_alpha = 2.0
        cd_beta = 2.0
        # calculate H-Measure using the python implementation
        scores = pd.read_csv(score_path)
        scores = {'class_0': scores['score_0'].dropna(),
                  'class_1': scores['score_1'].dropna()}
        crd = CostRatioDensity(beta(cd_alpha, cd_beta))
        hm = HMeasure(cost_distribution=crd)
        _, _, h = hm.h_measure(scores)

        # calculate H-measure using the Rust implementation
        hm = pyhmeasure.PyHmeasure(cd_alpha, cd_beta, None, None, score_path)

        np.testing.assert_almost_equal(h, hm.h, decimal=9)
