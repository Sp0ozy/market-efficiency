"""
core/baselines.py — dumb baselines that Elo must clearly beat.
"""

import numpy as np


def uniform_predictions(n: int) -> np.ndarray:
    """n rows of (1/3, 1/3, 1/3). The floor -- log_loss(uniform) == ln(3)."""
    assert n > 0, "n must be positive"
    return np.full((n, 3), 1.0 / 3.0)


def base_rate_predictions(outcomes: np.ndarray) -> np.ndarray:
    outcomes = np.asarray(outcomes)
    assert outcomes.ndim == 1 and len(outcomes) > 0, "outcomes must be non-empty (n,)"
    assert np.isin(outcomes, [0, 1, 2]).all(), "outcomes must be in {0, 1, 2}"

    counts = np.bincount(outcomes, minlength=3)
    rate = counts / counts.sum()
    return np.tile(rate, (len(outcomes), 1))


def home_always_predictions(n: int, eps: float = 1e-3) -> np.ndarray:
    assert n > 0, "n must be positive"
    assert 0 < eps < 0.5, "eps must be in (0, 0.5)"
    row = np.array([1.0 - 2.0 * eps, eps, eps])
    return np.tile(row, (n, 1))