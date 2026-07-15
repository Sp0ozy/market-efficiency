"""
core/bias.py — favorite-longshot bias: bin by implied probability, test the
calibration residual with confidence intervals.
"""

import numpy as np

from core.scoring import calibration_curve


def bias_table(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10, z: float = 1.96
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin calibration residual with a Wilson score confidence interval.

    Returns, one entry per non-empty bin:
      mean_predicted -- average predicted probability in the bin
      empirical_freq -- realized hit rate in the bin
      ci_lo, ci_hi   -- Wilson score interval on empirical_freq at level z
      residual       -- empirical_freq - mean_predicted
      significant    -- True where mean_predicted falls outside [ci_lo, ci_hi]
      counts         -- number of pooled pairs in the bin
    """
    assert z > 0, "z must be positive"
    mean_predicted, empirical_freq, counts = calibration_curve(probs, outcomes, n_bins=n_bins)

    n = counts.astype(float)
    p = empirical_freq
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    ci_lo = center - margin
    ci_hi = center + margin

    residual = empirical_freq - mean_predicted
    significant = (mean_predicted < ci_lo) | (mean_predicted > ci_hi)

    return mean_predicted, empirical_freq, ci_lo, ci_hi, residual, significant, counts

