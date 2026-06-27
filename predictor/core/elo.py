"""Naive O(n^2) Elo implementation.

This is a faithful port of lccn_predictor's ``app/core/elo.py``. It is the
**reference oracle**: correct but slow. The fast FFT path (``fft.py``) is what
the service actually uses; ``tests/test_elo.py`` checks the two agree.

LeetCode's Elo, in brief:
  * expected win rate of A vs B: ``1 / (1 + 10 ** ((R_B - R_A) / 400))``
  * expected rank of i: ``0.5 + sum_j E(R_j beats R_i)``
  * mean rank: ``sqrt(expected_rank * actual_rank)``
  * expected rating: binary-search the rating whose expected rank == mean rank
  * delta: ``(expected_rating - rating) * coef(k)`` where ``k`` = contests attended
  * coef shrinks toward 2/9 as k grows (veterans move less per contest)

Numba is OPTIONAL: if it isn't installed we fall back to a no-op ``jit`` so the
plain NumPy code still runs (just slower).
"""
from functools import lru_cache
from typing import Final

import numpy as np

try:  # numba is optional — only accelerates this oracle
    from numba import jit
except Exception:  # pragma: no cover - exercised only when numba is absent

    def jit(*args, **kwargs):
        def _decorator(func):
            return func

        # support both @jit and @jit(nopython=True, ...)
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _decorator


@lru_cache
def pre_sum_of_sigma(k: int) -> float:
    """Cached partial sum of the geometric series (5/7)^i for i in [0, k]."""
    if k < 0:
        raise ValueError(f"{k=}, pre_sum's index less than zero!")
    return (5 / 7) ** k + pre_sum_of_sigma(k - 1) if k >= 1 else 1


@lru_cache
def adjustment_for_delta_coefficient(k: int) -> float:
    """Delta coefficient for a player who has attended ``k`` rated contests.

    Equivalent to ``1 / (1 + sum((5/7)**i for i in range(k + 1)))``.
    Converges to ~2/9 for large k.
    """
    return 1 / (1 + pre_sum_of_sigma(k)) if k <= 100 else 2 / 9


def delta_coefficients(ks: np.ndarray) -> np.ndarray:
    """Vectorised delta coefficient over an array of attended-contest counts."""
    vectorized_func = np.vectorize(adjustment_for_delta_coefficient)
    return vectorized_func(ks)


@jit(nopython=True, fastmath=True)
def expected_win_rate(vector: np.ndarray, scalar: float) -> np.ndarray:
    """Elo expected win rate of ``scalar`` rating vs every rating in ``vector``."""
    return 1 / (1 + np.power(10, (scalar - vector) / 400))


@jit(nopython=True, fastmath=True)
def binary_search_expected_rating(mean_rank: float, all_rating: np.ndarray) -> float:
    """Find the rating whose expected rank equals ``mean_rank`` (precision 0.01)."""
    target = mean_rank - 1
    lo, hi = 0.0, 4000.0
    max_iteration = 25
    precision: Final[float] = 0.01
    mid = lo
    while hi - lo > precision and max_iteration >= 0:
        mid = lo + (hi - lo) / 2
        if np.sum(expected_win_rate(all_rating, mid)) < target:
            hi = mid
        else:
            lo = mid
        max_iteration -= 1
    return mid


@jit(nopython=True, fastmath=True)
def get_expected_rating(rank: int, rating: float, all_rating: np.ndarray) -> float:
    expected_rank = np.sum(expected_win_rate(all_rating, rating)) + 0.5
    mean_rank = np.sqrt(expected_rank * rank)
    return binary_search_expected_rating(mean_rank, all_rating)


def elo_delta(ranks: np.ndarray, ratings: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """Rating deltas for every participant. Inputs are parallel arrays sorted any way.

    :param ranks: actual finishing positions (1-based)
    :param ratings: current (pre-contest) ratings
    :param ks: attended-contest counts (the Elo K-factor index)
    """
    expected_ratings = []
    for i in range(len(ranks)):
        expected_ratings.append(get_expected_rating(ranks[i], ratings[i], ratings))
    return (np.array(expected_ratings) - ratings) * delta_coefficients(ks)
