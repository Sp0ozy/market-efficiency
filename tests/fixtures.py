"""
PURELY AI GENERATED FILE

tests/fixtures.py — fake Contract-A predictions tables for testing before
the real model or market adapter exist.

Contract A (the predictions table), emitted by every model and by the market:
    date | div | season | home | away | p_home | p_draw | p_away | outcome

Two independent fake tables (`fake_market_table`, `fake_model_table`) describe
the SAME matches (same date/div/season/home/away/outcome) but carry
independently random probabilities, so tests can compare "model" against
"market" without either real pipeline existing yet.

"""

import numpy as np
import pandas as pd

N = 500
_SEED_MATCHES = 0
_SEED_MARKET_PROBS = 1
_SEED_MODEL_PROBS = 2

_OUTCOME_TO_INT = {"H": 0, "D": 1, "A": 2}

_TEAMS = [
    "Arsenal", "Aston Villa", "Chelsea", "Everton", "Fulham",
    "Liverpool", "Man City", "Man United", "Newcastle", "Tottenham",
]


def _random_normalized_probs(rng: np.random.Generator, n: int) -> np.ndarray:
    """n draws of 3 positive numbers normalized to sum to 1 per row."""
    raw = rng.uniform(0.01, 1.0, size=(n, 3))
    return raw / raw.sum(axis=1, keepdims=True)


def _base_matches() -> pd.DataFrame:
    """The shared match skeleton: date, div, season, home, away, outcome."""
    rng = np.random.default_rng(_SEED_MATCHES)
    dates = pd.date_range("2020-08-01", periods=N, freq="D")

    home = rng.choice(_TEAMS, size=N)
    away = np.empty(N, dtype=object)
    for i in range(N):
        choices = [t for t in _TEAMS if t != home[i]]
        away[i] = rng.choice(choices)

    outcome_idx = rng.integers(0, 3, size=N)
    outcome = np.array(["H", "D", "A"])[outcome_idx]

    season = [f"{d.year % 100:02d}{(d.year + 1) % 100:02d}" for d in dates]

    df = pd.DataFrame({
        "date": dates,
        "div": "E0",
        "season": season,
        "home": home,
        "away": away,
        "outcome": outcome,
    })
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def fake_market_table() -> pd.DataFrame:
    """500-row fake Contract-A table standing in for the market adapter."""
    base = _base_matches()
    rng = np.random.default_rng(_SEED_MARKET_PROBS)
    probs = _random_normalized_probs(rng, len(base))
    return base.assign(p_home=probs[:, 0], p_draw=probs[:, 1], p_away=probs[:, 2])[
        ["date", "div", "season", "home", "away", "p_home", "p_draw", "p_away", "outcome"]
    ]


def fake_model_table() -> pd.DataFrame:
    """500-row fake Contract-A table standing in for a model's predictions."""
    base = _base_matches()
    rng = np.random.default_rng(_SEED_MODEL_PROBS)
    probs = _random_normalized_probs(rng, len(base))
    return base.assign(p_home=probs[:, 0], p_draw=probs[:, 1], p_away=probs[:, 2])[
        ["date", "div", "season", "home", "away", "p_home", "p_draw", "p_away", "outcome"]
    ]


def to_arrays(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Adapt a Contract-A table to Contract B arrays: probs (n,3), outcomes (n,)."""
    probs = table[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    outcomes = table["outcome"].map(_OUTCOME_TO_INT).to_numpy()
    return probs, outcomes

