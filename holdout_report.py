import numpy as np
import pandas as pd

from adapters.football import (
    load_matches,
    season_start,
    pinnacle_close,
    pinnacle_early,
    bet365_close,
    bet365_early,
)
from core.baselines import home_always_predictions, uniform_predictions
from core.scoring import brier, log_loss
from models.elo import run_elo

_OUTCOME_TO_INT = {"H": 0, "D": 1, "A": 2}
_KEYS = ["date", "home", "away"]


def _outcomes(df: pd.DataFrame) -> np.ndarray:
    return df["outcome"].map(_OUTCOME_TO_INT).to_numpy()


def main() -> None:
    matches = load_matches()
    holdout_start = season_start(matches, "1920")
    train_outcomes = _outcomes(matches[matches["date"] < holdout_start])

    elo_preds, _ratings = run_elo(matches)
    sources = {
        "pinnacle_close": pinnacle_close(),
        "pinnacle_early": pinnacle_early(),
        "bet365_close": bet365_close(),
        "bet365_early": bet365_early(),
    }

    # Shared window: matches present in ALL four sources, restricted to the holdout.
    shared = sources["pinnacle_close"][_KEYS]
    for name in ("pinnacle_early", "bet365_close", "bet365_early"):
        shared = shared.merge(sources[name][_KEYS], on=_KEYS, how="inner")
    shared = shared[shared["date"] >= holdout_start]

    def _aligned(table: pd.DataFrame) -> pd.DataFrame:
        return (
            table.merge(shared, on=_KEYS, how="inner")
            .sort_values("date", kind="stable")
            .reset_index(drop=True)
        )

    elo_h = _aligned(elo_preds)
    aligned_sources = {name: _aligned(t) for name, t in sources.items()}

    y = _outcomes(elo_h)
    for name, t in aligned_sources.items():
        assert (y == _outcomes(t)).all(), f"{name} outcomes don't line up with elo"

    n = len(y)
    train_rate = np.bincount(train_outcomes, minlength=3) / len(train_outcomes)

    predictions = {
        "uniform": uniform_predictions(n),
        "base_rate (train-only)": np.tile(train_rate, (n, 1)),
        "home_always": home_always_predictions(n),
        "elo": elo_h[["p_home", "p_draw", "p_away"]].to_numpy(),
    }
    for name, t in aligned_sources.items():
        predictions[name] = t[["p_home", "p_draw", "p_away"]].to_numpy()

    report = pd.DataFrame(
        {name: {"log_loss": log_loss(p, y), "brier": brier(p, y)} for name, p in predictions.items()}
    ).T
    report.index.name = "model"

    print(f"holdout start (season 1920): {holdout_start.date()}")
    print(f"holdout window (shared across all 4 market sources): "
          f"{shared['date'].min().date()} -> {shared['date'].max().date()}  (n={n})")
    print(report.round(4).to_string())

    print()
    for name in ("pinnacle_close", "bet365_close"):
        corr = np.corrcoef(elo_h["p_home"], aligned_sources[name]["p_home"])[0, 1]
        print(f"corr(p_home elo, p_home {name}) = {corr:.3f}")

if __name__ == "__main__":
    main()