import pytest
import numpy as np
import pandas as pd
from scipy.stats import beta

import pyhmeasure # the python module exposing the Rust implementation of H-Measure

# a local python implementation of H-Measure for comparison in the test + utilities for generating test examples
from python_src.hmeasure.datagen import DataGenBinaryClassifierScores
from python_src.hmeasure.h_measure import HMeasure, CostRatioDensity

# the parameters of the cost distribution (a Beta distribution)
CD_ALPHA = 2.0
CD_BETA = 2.0

SMALL_GROUP = "3,600 scores"
LARGE_GROUP = "360,000 scores"


def generate_scores(c0_sample_size, c1_sample_size):
    # Generate dummy score data for class0 and class1 to be used by the tests
    class_params = {
        'class0_alpha': 2.0,
        'class0_beta': 6.0,
        'class1_alpha': 6.0,
        'class1_beta': 2.0
    }
    np.random.seed(13)  # the same scores on every run, so that benchmark runs are comparable
    dg_bcs = DataGenBinaryClassifierScores(class_params=class_params,
                                           c0_sample_size=c0_sample_size,
                                           c1_sample_size=c1_sample_size)
    score_samples = dg_bcs.generate_samples()
    return score_samples.get('class_0').tolist(), score_samples.get('class_1').tolist()


def write_scores(directory, c0scores, c1scores):
    df = pd.DataFrame({"score_0": pd.Series(c0scores), "score_1": pd.Series(c1scores)})
    path = str(directory / "scores.csv")
    df.to_csv(path)
    return path


@pytest.fixture(scope="module")
def scores():
    return generate_scores(2000, 1600)  # slightly imbalanced classification problem


@pytest.fixture(scope="module")
def score_path(scores, tmp_path_factory):
    return write_scores(tmp_path_factory.mktemp("scores"), *scores)


@pytest.fixture(scope="module")
def large_scores():
    return generate_scores(200000, 160000)


@pytest.fixture(scope="module")
def large_score_path(large_scores, tmp_path_factory):
    return write_scores(tmp_path_factory.mktemp("large_scores"), *large_scores)


class TestHmeasure:
    # Benchmarks: the python and Rust implementations of the same algorithm on the same scores, with the scores
    # either loaded from a csv file or passed in as lists.

    @pytest.mark.benchmark(group=SMALL_GROUP)
    def test_python_hmeasure(self, score_path, benchmark):
        benchmark(self.run_pure_python, score_path)

    def run_pure_python(self, score_path):
        # call the pure python implementation of H-Measure, loading the scores from file with pandas
        scores = pd.read_csv(score_path)
        scores = {'class_0': scores['score_0'].dropna(),
                  'class_1': scores['score_1'].dropna()}
        crd = CostRatioDensity(beta(CD_ALPHA, CD_BETA))
        hm = HMeasure(cost_distribution=crd)
        _ = hm.h_measure(scores)

    @pytest.mark.benchmark(group=SMALL_GROUP)
    def test_python_hmeasure_scores_passed_in(self, scores, benchmark):
        benchmark(self.run_pure_python_scores_passed_in, *scores)

    def run_pure_python_scores_passed_in(self, c0scores, c1scores):
        # call the pure python implementation of H-Measure, passing the scores in as lists
        crd = CostRatioDensity(beta(CD_ALPHA, CD_BETA))
        hm = HMeasure(cost_distribution=crd)
        _ = hm.h_measure({'class_0': c0scores, 'class_1': c1scores})

    @pytest.mark.benchmark(group=SMALL_GROUP)
    def test_rust_hmeasure(self, score_path, benchmark):
        benchmark(self.run_rust, score_path)

    def run_rust(self, score_path):
        # call the rust implementation of H-Measure exposed in python, loading the scores from file in Rust
        _ = pyhmeasure.PyHmeasure(CD_ALPHA, CD_BETA, None, None, score_path)

    @pytest.mark.benchmark(group=SMALL_GROUP)
    def test_rust_hmeasure_scores_passed_in(self, scores, benchmark):
        benchmark(self.run_rust_scores_passed_in, *scores)

    def run_rust_scores_passed_in(self, c0scores, c1scores):
        # call the rust implementation of H-Measure exposed in python, passing the scores in as lists
        _ = pyhmeasure.PyHmeasure(CD_ALPHA, CD_BETA, c0scores, c1scores, None)

    # Benchmarks: the same comparisons on 100 times as many scores.

    @pytest.mark.benchmark(group=LARGE_GROUP)
    def test_python_hmeasure_large(self, large_score_path, benchmark):
        benchmark(self.run_pure_python, large_score_path)

    @pytest.mark.benchmark(group=LARGE_GROUP)
    def test_python_hmeasure_large_scores_passed_in(self, large_scores, benchmark):
        benchmark(self.run_pure_python_scores_passed_in, *large_scores)

    @pytest.mark.benchmark(group=LARGE_GROUP)
    def test_rust_hmeasure_large(self, large_score_path, benchmark):
        benchmark(self.run_rust, large_score_path)

    @pytest.mark.benchmark(group=LARGE_GROUP)
    def test_rust_hmeasure_large_scores_passed_in(self, large_scores, benchmark):
        benchmark(self.run_rust_scores_passed_in, *large_scores)

    # Tests

    def test_assert_almost_equal(self, score_path):
        # calculate H-Measure using the python implementation
        scores = pd.read_csv(score_path)
        scores = {'class_0': scores['score_0'].dropna(),
                  'class_1': scores['score_1'].dropna()}
        crd = CostRatioDensity(beta(CD_ALPHA, CD_BETA))
        hm = HMeasure(cost_distribution=crd)
        _, _, h = hm.h_measure(scores)

        # calculate H-measure using the Rust implementation
        hm = pyhmeasure.PyHmeasure(CD_ALPHA, CD_BETA, None, None, score_path)

        np.testing.assert_almost_equal(h, hm.h, decimal=12)

    def test_assert_almost_equal_large(self, large_scores):
        crd = CostRatioDensity(beta(CD_ALPHA, CD_BETA))
        hm = HMeasure(cost_distribution=crd)
        _, _, h = hm.h_measure({'class_0': large_scores[0], 'class_1': large_scores[1]})

        hm = pyhmeasure.PyHmeasure(CD_ALPHA, CD_BETA, large_scores[0], large_scores[1], None)

        np.testing.assert_almost_equal(h, hm.h, decimal=12)

    def test_python_numerical_integration_matches_closed_form(self, scores):
        # The python implementation integrates a Beta(a, b) cost distribution in closed form when it is created
        # as beta(a, b), and numerically otherwise: creating it with keyword arguments takes the numerical route.
        scores = {'class_0': scores[0], 'class_1': scores[1]}
        closed_form = CostRatioDensity(beta(3.3, 1.7))
        numerical = CostRatioDensity(beta(a=3.3, b=1.7))
        assert closed_form.beta_params == (3.3, 1.7)
        assert numerical.beta_params is None

        _, _, h_closed_form = HMeasure(cost_distribution=closed_form).h_measure(scores)
        _, _, h_numerical = HMeasure(cost_distribution=numerical).h_measure(scores)

        np.testing.assert_almost_equal(h_closed_form, h_numerical, decimal=9)

    def test_scores_from_file_and_passed_in_agree(self, scores, score_path):
        from_file = pyhmeasure.PyHmeasure(CD_ALPHA, CD_BETA, None, None, score_path)
        passed_in = pyhmeasure.PyHmeasure(CD_ALPHA, CD_BETA, scores[0], scores[1], None)

        np.testing.assert_almost_equal(from_file.h, passed_in.h, decimal=12)

    def test_numpy_arrays_are_accepted(self, scores):
        from_lists = pyhmeasure.PyHmeasure(CD_ALPHA, CD_BETA, scores[0], scores[1], None)
        from_arrays = pyhmeasure.PyHmeasure(CD_ALPHA, CD_BETA, np.array(scores[0]), np.array(scores[1]), None)

        assert from_lists.h == from_arrays.h

    @pytest.mark.parametrize("cd_alpha, cd_beta, c0scores, c1scores, message", [
        (0.0, 2.0, [0.1, 0.2], [0.8, 0.9], "Beta parameters"),
        (2.0, 2.0, [], [0.8, 0.9], "class0 has no scores"),
        (2.0, 2.0, [0.1, 0.2], [], "class1 has no scores"),
        (2.0, 2.0, [0.1, float("nan")], [0.8, 0.9], "NaN score"),
    ])
    def test_invalid_inputs_raise_value_error(self, cd_alpha, cd_beta, c0scores, c1scores, message):
        with pytest.raises(ValueError, match=message):
            pyhmeasure.PyHmeasure(cd_alpha, cd_beta, c0scores, c1scores, None)
