"""
core/devig.py — decimal odds to implied probabilities, multiplicative method.
Known limitation multiplicative de-vig inflates longshot probabilities relative to fair value.
"""

import numpy as np


def devig_multiplicative(odds: np.ndarray) -> np.ndarray:
    odds = np.asarray(odds, dtype=float)
    assert odds.ndim == 2 and odds.shape[1] == 3, (
        f"odds must be shape (n, 3), got {odds.shape}"
    )
    assert (odds > 1.0).all(), "all decimal odds must be > 1.0"

    raw = 1.0 / odds
    return raw / raw.sum(axis=1, keepdims=True)