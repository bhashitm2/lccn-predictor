"""Public prediction entry point used by the service layer.

``predict`` takes the three parallel arrays describing a finished contest and
returns ``(deltas, new_ratings)``. It defaults to the fast FFT path and falls
back to the naive oracle for very small fields (where building an 800k-point
convolution would be wasteful and the O(n^2) cost is negligible).
"""
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from predictor.core.elo import elo_delta
from predictor.core.fft import fft_delta

# Below this many participants the naive path is both fast enough and avoids the
# fixed cost of the large FFT kernel.
_FFT_MIN_PARTICIPANTS = 500


@dataclass
class PredictionInput:
    ranks: Sequence[int]
    ratings: Sequence[float]
    attended_counts: Sequence[int]


def predict(
    ranks: Sequence[int],
    ratings: Sequence[float],
    attended_counts: Sequence[int],
    *,
    method: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rating deltas and new ratings for a contest.

    :param ranks: actual finishing positions (1-based), one per participant
    :param ratings: pre-contest ratings, same order
    :param attended_counts: number of rated contests previously attended
    :param method: ``"auto"`` (default), ``"fft"`` or ``"naive"``
    :returns: ``(deltas, new_ratings)`` as float numpy arrays
    """
    ranks_arr = np.asarray(ranks, dtype=np.float64)
    ratings_arr = np.asarray(ratings, dtype=np.float64)
    ks_arr = np.asarray(attended_counts, dtype=np.int64)

    n = len(ranks_arr)
    if not (len(ratings_arr) == len(ks_arr) == n):
        raise ValueError("ranks, ratings and attended_counts must be equal length")
    if n == 0:
        return np.array([]), np.array([])

    if method == "naive" or (method == "auto" and n < _FFT_MIN_PARTICIPANTS):
        deltas = elo_delta(ranks_arr, ratings_arr, ks_arr)
    else:
        deltas = fft_delta(ranks_arr, ratings_arr, ks_arr)

    new_ratings = ratings_arr + deltas
    return deltas, new_ratings
