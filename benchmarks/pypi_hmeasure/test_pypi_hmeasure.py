"""
Benchmarks pyhmeasure against hmeasure on PyPI (https://pypi.org/project/hmeasure/), a Python
translation of the R hmeasure package, and checks that the two compute the same H-measure.

Both packages are given the same inputs: a NumPy array of 0/1 labels and a NumPy array of scores.
pyhmeasure takes the scores of each class separately, so its timing includes splitting the scores
by label.
"""
import numpy as np
import pytest
from hmeasure import h_score

import pyhmeasure

SIZES = [3_600, 36_000, 360_000, 3_600_000]
# rounds per benchmark, fewer for the larger sizes to keep the run to about a minute
ROUNDS = {3_600: 100, 36_000: 30, 360_000: 8, 3_600_000: 3}


def pyhmeasure_h(y, scores, alpha, beta):
    return pyhmeasure.PyHmeasure(alpha, beta, scores[y == 0], scores[y == 1]).h


def make_scores(n, seed=13):
    """n scores from a classifier that separates the classes fairly well, with 5 class 0 for every 4 class 1."""
    rng = np.random.default_rng(seed)
    n0 = n * 5 // 9
    n1 = n - n0
    y = np.concatenate([np.zeros(n0, dtype=np.int64), np.ones(n1, dtype=np.int64)])
    scores = np.concatenate([rng.beta(2.0, 6.0, n0), rng.beta(6.0, 2.0, n1)])
    return y, scores


# The two packages compute the same H-measure


def random_case(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.choice([4, 10, 50, 500, 5000]))
    y = (rng.random(n) < rng.uniform(0.1, 0.9)).astype(np.int64)
    y[:2] = [0, 1]  # both classes present
    separation = rng.uniform(0.0, 1.0)
    scores = rng.beta(2.0 + 3.0 * y * separation, 2.0 + 3.0 * (1 - y) * separation)
    if seed % 3 == 0:
        scores = np.round(scores, 1)  # tied scores
    return y, scores


@pytest.mark.parametrize("seed", range(60))
def test_same_h_measure_with_beta_2_2_cost_distribution(seed):
    # hmeasure's cost distribution is Beta(2, 1 + 1/severity_ratio): severity_ratio=1 gives Beta(2, 2)
    y, scores = random_case(seed)
    np.testing.assert_allclose(
        h_score(y, scores, severity_ratio=1.0), pyhmeasure_h(y, scores, 2.0, 2.0), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("seed", range(60))
def test_same_h_measure_with_hmeasure_default_cost_distribution(seed):
    # hmeasure's default severity ratio is n1/n0, giving Beta(2, 1 + n0/n1)
    y, scores = random_case(seed)
    n1 = y.sum()
    n0 = len(y) - n1
    np.testing.assert_allclose(
        h_score(y, scores), pyhmeasure_h(y, scores, 2.0, 1.0 + n0 / n1), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("n", SIZES)
def test_same_h_measure_on_benchmark_data(n):
    y, scores = make_scores(n)
    np.testing.assert_allclose(
        h_score(y, scores, severity_ratio=1.0), pyhmeasure_h(y, scores, 2.0, 2.0), rtol=0, atol=1e-12
    )


# Benchmarks, with a Beta(2, 2) cost distribution


@pytest.fixture(scope="module", params=SIZES, ids=lambda n: f"{n:,}")
def benchmark_data(request):
    return request.param, make_scores(request.param)


def test_pypi_hmeasure(benchmark, benchmark_data):
    n, (y, scores) = benchmark_data
    benchmark.group = f"{n:,} scores"
    benchmark.pedantic(h_score, args=(y, scores), kwargs={"severity_ratio": 1.0}, rounds=ROUNDS[n], warmup_rounds=1)


def test_pyhmeasure(benchmark, benchmark_data):
    n, (y, scores) = benchmark_data
    benchmark.group = f"{n:,} scores"
    benchmark.pedantic(pyhmeasure_h, args=(y, scores, 2.0, 2.0), rounds=ROUNDS[n], warmup_rounds=1)
