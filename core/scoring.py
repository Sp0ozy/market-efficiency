import numpy as np

_EPS = 1e-15


def _validate(probs: np.ndarray, outcomes: np.ndarray) -> None:
    probs = np.asarray(probs)
    outcomes = np.asarray(outcomes)
    assert probs.ndim == 2 and probs.shape[1] == 3, (
        f"probs must be shape (n, 3), got {probs.shape}"
    )
    assert outcomes.ndim == 1 and outcomes.shape[0] == probs.shape[0], (
        f"outcomes must be shape (n,) matching probs, got {outcomes.shape} vs {probs.shape}"
    )
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6), (
        "probs rows must sum to 1.0"
    )
    assert np.isin(outcomes, [0, 1, 2]).all(), (
        "outcomes must be in {0, 1, 2}"
    )


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean negative log-likelihood of the realized outcome under `probs`."""
    _validate(probs, outcomes)
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes)
    p_true = probs[np.arange(len(outcomes)), outcomes]
    p_true = np.clip(p_true, _EPS, 1.0)
    return float(-np.mean(np.log(p_true)))


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error between `probs` and the one-hot realized outcome."""
    _validate(probs, outcomes)
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(outcomes)), outcomes] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def calibration_curve(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin predicted probabilities and compare to empirical hit frequency."""
    _validate(probs, outcomes)
    assert n_bins > 0, "n_bins must be positive"
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes)

    n = probs.shape[0]
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), outcomes] = 1.0

    pooled_probs = probs.reshape(-1)
    pooled_hits = onehot.reshape(-1)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(pooled_probs, edges[1:-1], right=True), 0, n_bins - 1)

    mean_predicted = []
    empirical_freq = []
    counts = []
    for b in range(n_bins):
        mask = bin_idx == b
        c = int(mask.sum())
        if c == 0:
            continue
        mean_predicted.append(float(pooled_probs[mask].mean()))
        empirical_freq.append(float(pooled_hits[mask].mean()))
        counts.append(c)

    return (
        np.array(mean_predicted),
        np.array(empirical_freq),
        np.array(counts, dtype=int),
    )