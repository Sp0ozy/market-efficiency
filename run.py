from pathlib import Path

import numpy as np
import pandas as pd

from adapters.football import load_matches, market_table
from core.baselines import base_rate_predictions, home_always_predictions, uniform_predictions
from core.scoring import brier, log_loss
from models.elo import run_elo

REPORTS = Path("reports")
_OUTCOME_TO_INT = {"H": 0, "D": 1, "A": 2}


def _to_arrays(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    probs = table[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    outcomes = table["outcome"].map(_OUTCOME_TO_INT).to_numpy()
    return probs, outcomes


def main() -> None:
    matches = load_matches()
    elo_preds, _ratings = run_elo(matches)
    market = market_table()

    eval_elo = elo_preds.merge(
        market[["date", "home", "away"]], on=["date", "home", "away"], how="inner"
    )

    p_elo, y = _to_arrays(eval_elo)
    p_mkt, y_mkt = _to_arrays(market)
    assert (y == y_mkt).all(), "eval-window outcomes must line up between elo and market"

    n = len(y)
    predictions = {
        "uniform": uniform_predictions(n),
        "base_rate": base_rate_predictions(y),
        "home_always": home_always_predictions(n),
        "elo": p_elo,
        "pinnacle_close": p_mkt,
    }

    report = pd.DataFrame(
        {name: {"log_loss": log_loss(p, y), "brier": brier(p, y)} for name, p in predictions.items()}
    ).T
    report.index.name = "model"

    REPORTS.mkdir(exist_ok=True)
    report.to_csv(REPORTS / "scores.csv")

    print(f"eval window: {market['date'].min().date()} -> {market['date'].max().date()}  (n={n})")
    print(report.round(4).to_string())

    corr = np.corrcoef(p_elo[:, 0], p_mkt[:, 0])[0, 1]
    print(f"\ncorr(p_home elo, p_home market) = {corr:.3f}")


if __name__ == "__main__":
    main()