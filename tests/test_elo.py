"""Algorithm correctness: the FFT path must match the naive oracle.

Reproduces lccn_predictor's precision claim — per-participant delta error within
0.05 between the O(n^2) reference and the O(n log n) FFT implementation.
"""
import numpy as np
import pytest

from predictor.core.elo import (
    adjustment_for_delta_coefficient,
    elo_delta,
)
from predictor.core.engine import predict
from predictor.core.fft import fft_delta


def _make_contest(n: int, seed: int = 0):
    """Random but plausible contest: ratings ~N(1700, 350), ranks = sorted order
    with a bit of noise so rank and rating aren't perfectly correlated."""
    rng = np.random.default_rng(seed)
    ratings = np.clip(rng.normal(1700, 350, n), 800, 3600)
    noise = rng.normal(0, 200, n)
    order = np.argsort(-(ratings + noise))  # stronger (after noise) ranks better
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1)
    ks = rng.integers(0, 60, n)
    return ranks, ratings, ks


@pytest.mark.parametrize("n,seed", [(50, 1), (300, 2), (1500, 3)])
def test_fft_matches_naive(n, seed):
    ranks, ratings, ks = _make_contest(n, seed)
    naive = elo_delta(ranks, ratings, ks)
    fast = fft_delta(ranks, ratings, ks)
    assert np.max(np.abs(naive - fast)) < 0.05


def test_delta_coefficient_converges_to_two_ninths():
    # New player moves at coef 1/2 (k=0): 1/(1 + 1) = 0.5
    assert adjustment_for_delta_coefficient(0) == pytest.approx(0.5)
    # Veteran converges toward 2/9
    assert adjustment_for_delta_coefficient(200) == pytest.approx(2 / 9)
    assert adjustment_for_delta_coefficient(80) == pytest.approx(2 / 9, abs=1e-3)


def test_monotonic_and_directional_for_uniform_newcomers():
    """With an otherwise identical field (all 1500, k=0), delta must depend
    monotonically on rank: finishing better gives a strictly larger delta, the
    winner gains and last place loses.

    Note: LeetCode's Elo is NOT zero-sum (rank->expected-rating is nonlinear via
    the geometric-mean-of-ranks step), so deltas do not cancel — that is expected
    behaviour, faithfully reproduced from the reference implementation.
    """
    n = 200
    ranks = np.arange(1, n + 1, dtype=np.float64)
    ratings = np.full(n, 1500.0)
    ks = np.zeros(n, dtype=np.int64)
    deltas, new_ratings = predict(ranks, ratings, ks, method="naive")
    # strictly decreasing in rank (rank 1 best delta ... rank n worst)
    assert np.all(np.diff(deltas) < 0)
    # winner gains, last place loses
    assert deltas[0] > 0
    assert deltas[-1] < 0
    assert new_ratings[0] > 1500.0 > new_ratings[-1]


def test_engine_auto_picks_consistent_results():
    ranks, ratings, ks = _make_contest(800, seed=7)
    d_fft, _ = predict(ranks, ratings, ks, method="fft")
    d_naive, _ = predict(ranks, ratings, ks, method="naive")
    assert np.max(np.abs(d_fft - d_naive)) < 0.05


def test_engine_handles_empty():
    deltas, new_ratings = predict([], [], [])
    assert len(deltas) == 0 and len(new_ratings) == 0
