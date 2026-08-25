"""
core/staking.py — Kelly criterion staking and bankroll simulation.

Domain-agnostic: arrays only. `odds` throughout means the market's actual
quoted decimal odds (vig included), not a de-vigged fair price. Any edge
computed here already nets out the bookmaker's margin, since it is priced
against what you could really place a bet at.
"""

import numpy as np


def kelly_fraction(probs: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Full-Kelly stake fraction for each (probability, odds) pair.

    f* = (p*b - (1-p)) / b, where b = odds - 1 (net odds). Negative-edge
    pairs are clipped to 0 (no bet), never a negative/short stake.
    """
    probs = np.asarray(probs, dtype=float)
    odds = np.asarray(odds, dtype=float)
    assert probs.shape == odds.shape, "probs and odds must have the same shape"
    assert ((probs >= 0) & (probs <= 1)).all(), "probs must be in [0, 1]"
    assert (odds > 1.0).all(), "odds must be > 1.0"

    b = odds - 1.0
    f = (probs * b - (1.0 - probs)) / b
    return np.clip(f, 0.0, None)


def best_edge_bets(
    model_probs: np.ndarray, odds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row, pick the single outcome column with the largest positive edge.

    edge = model_probs * odds - 1 (expected return per unit staked at the
    quoted price). Returns (chosen_col, stake_fraction, edge_of_chosen):
    chosen_col is -1 and stake_fraction is 0 wherever no column has a
    positive edge (no bet placed on that row).
    """
    model_probs = np.asarray(model_probs, dtype=float)
    odds = np.asarray(odds, dtype=float)
    assert model_probs.shape == odds.shape, "model_probs and odds must have the same shape"

    edge = model_probs * odds - 1.0
    chosen_col = np.argmax(edge, axis=1)
    n = model_probs.shape[0]
    rows = np.arange(n)

    best_edge = edge[rows, chosen_col]
    best_p = model_probs[rows, chosen_col]
    best_odds = odds[rows, chosen_col]

    has_edge = best_edge > 0
    stake_fraction = np.where(has_edge, kelly_fraction(best_p, best_odds), 0.0)
    chosen_col = np.where(has_edge, chosen_col, -1)

    return chosen_col, stake_fraction, best_edge


def bankroll_simulation(
    stake_fraction: np.ndarray,
    odds: np.ndarray,
    hit: np.ndarray,
    starting_bankroll: float = 1.0,
    kelly_multiplier: float = 1.0,
) -> np.ndarray:
    """Sequential bankroll path: stake stake_fraction*kelly_multiplier of the
    current bankroll on each row, at `odds`, winning iff `hit`.

    Returns the bankroll trajectory, length n+1 (index 0 = starting_bankroll).
    One equity curve is one sample path, not evidence — state this wherever
    it is shown.
    """
    stake_fraction = np.asarray(stake_fraction, dtype=float) * kelly_multiplier
    odds = np.asarray(odds, dtype=float)
    hit = np.asarray(hit, dtype=bool)
    n = len(stake_fraction)
    assert len(odds) == n and len(hit) == n, "stake_fraction, odds, hit must be the same length"
    assert ((stake_fraction >= 0) & (stake_fraction <= 1)).all(), "stake_fraction must be in [0, 1]"

    bankroll = np.empty(n + 1)
    bankroll[0] = starting_bankroll
    for i in range(n):
        stake = stake_fraction[i] * bankroll[i]
        payout = stake * (odds[i] - 1.0) if hit[i] else -stake
        bankroll[i + 1] = bankroll[i] + payout
    return bankroll
