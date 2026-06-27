"""FFT-accelerated Elo — the path the service actually uses.

Faithful port of lccn_predictor's ``app/core/fft.py``. The naive oracle in
``elo.py`` is O(n^2): for each of n players it sums win-rates against all n
players. Here we instead:

1. Bin every player's rating into a histogram ``g`` at 0.01 resolution
   (``EXPAND_SIZE = 100``).
2. Pre-compute a logistic kernel ``f``.
3. ``fftconvolve(f, g)`` yields, for *every* rating value at once, the sum of
   win-rates against the whole field -> expected ranks in O(n log n).

After the single convolution, each player's expected rating is a binary search
over the pre-computed array. <0.25s for ~25k-40k participants.
"""
from typing import Final

import numpy as np
from scipy.signal import fftconvolve

from predictor.core.elo import delta_coefficients

EXPAND_SIZE: Final[int] = 100          # rating resolution = 1/100 = 0.01
MAX_RATING: Final[int] = 4000 * EXPAND_SIZE


def pre_calc_convolution(old_rating: np.ndarray) -> np.ndarray:
    """Pre-compute the win-rate convolution over the whole rating axis (once)."""
    f = 1 / (
        1 + np.power(10, np.arange(-MAX_RATING, MAX_RATING + 1) / (400 * EXPAND_SIZE))
    )
    g = np.bincount(np.round(old_rating * EXPAND_SIZE).astype(int))
    convolution = fftconvolve(f, g, mode="full")
    convolution = convolution[: 2 * MAX_RATING + 1]
    return convolution


def get_expected_rank(convolution: np.ndarray, x: int) -> float:
    return convolution[x + MAX_RATING] + 0.5


def get_equation_left(convolution: np.ndarray, x: int) -> float:
    return convolution[x + MAX_RATING] + 1


def binary_search_expected_rating(convolution: np.ndarray, mean_rank: float) -> int:
    lo, hi = 0, MAX_RATING
    mid = 0
    while lo < hi:
        mid = (lo + hi) // 2
        if get_equation_left(convolution, mid) < mean_rank:
            hi = mid
        else:
            lo = mid + 1
    return mid


def get_expected_rating(rank: int, rating: float, convolution: np.ndarray) -> float:
    expected_rank = get_expected_rank(convolution, round(rating * EXPAND_SIZE))
    mean_rank = np.sqrt(expected_rank * rank)
    return binary_search_expected_rating(convolution, mean_rank) / EXPAND_SIZE


def fft_delta(ranks: np.ndarray, ratings: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """Rating deltas for every participant, FFT-accelerated. Same contract as
    :func:`predictor.core.elo.elo_delta`."""
    convolution = pre_calc_convolution(ratings)
    expected_ratings = []
    for i in range(len(ranks)):
        expected_ratings.append(get_expected_rating(ranks[i], ratings[i], convolution))
    return (np.array(expected_ratings) - ratings) * delta_coefficients(ks)
