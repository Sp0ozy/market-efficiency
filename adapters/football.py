"""
adapters/football.py — clean.parquet -> the predictions-table contract.

Loads the cleaned E0 match data and exposes two views:
  load_matches()   the bare fixture list every model walks forward over:
                   date | div | season | home | away | outcome
  market_table()   the market's own Contract A entry, built by de-vigging
                   Pinnacle CLOSING odds (PSCH/PSCD/PSCA). Rows where the
                   closing line isn't available are dropped -- exactly what
                   defines the evaluation window.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from core.devig import devig_multiplicative

CLEAN = Path("data/clean.parquet")

_RENAME = {
    "Div": "div",
    "Date": "date",
    "HomeTeam": "home",
    "AwayTeam": "away",
    "FTR": "outcome",
}


def load_matches(path: Path = CLEAN) -> pd.DataFrame:
    """The list every model walks forward over, without probabilities."""
    df = pd.read_parquet(path)
    df = df.rename(columns=_RENAME)
    df = df[["date", "div", "season", "home", "away", "outcome"]].copy()
    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    assert df["date"].is_monotonic_increasing
    assert df["outcome"].isin(["H", "D", "A"]).all()
    return df


def market_table(path: Path = CLEAN) -> pd.DataFrame:
    """The dataframe including market probabilities, built by de-vigging Pinnacle CLOSING odds."""
    df = pd.read_parquet(path)
    df = df.rename(columns=_RENAME)
    df = df.dropna(subset=["PSCH", "PSCD", "PSCA"])

    odds = df[["PSCH", "PSCD", "PSCA"]].to_numpy(dtype=float)
    probs = devig_multiplicative(odds)

    out = df[["date", "div", "season", "home", "away", "outcome"]].copy()
    out["p_home"] = probs[:, 0]
    out["p_draw"] = probs[:, 1]
    out["p_away"] = probs[:, 2]
    out = out[["date", "div", "season", "home", "away", "p_home", "p_draw", "p_away", "outcome"]]
    out = out.sort_values("date", kind="stable").reset_index(drop=True)

    assert np.allclose(out[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0)
    assert out["date"].is_monotonic_increasing
    return out