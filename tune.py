"""
tune.py — grid-search K and HFA on the training window only, refitting the
draw-rate curve to match each combination.

CLAUDE.md hard rule #3: the holdout is sacred. All tuning happens on
2010-2019; 2019 onward is touched exactly once, at the end. This script
never reads or scores anything from TRAIN_CUTOFF onward.

Run: python tune.py
"""

import numpy as np
import pandas as pd

from adapters.football import load_matches
from core.scoring import log_loss
from models.elo import fit_draw_model, run_elo

TRAIN_CUTOFF = pd.Timestamp("2019-01-01")  # holdout starts here; never touched by tuning

K_GRID = [10, 15, 20, 25, 30, 40, 50]
HFA_GRID = [0, 50, 75, 100, 125, 150]

_OUTCOME_TO_INT = {"H": 0, "D": 1, "A": 2}


def _outcomes(df: pd.DataFrame) -> np.ndarray:
    return df["outcome"].map(_OUTCOME_TO_INT).to_numpy()


def main() -> None:
    matches = load_matches()
    train = matches[matches["date"] < TRAIN_CUTOFF]
    assert train["date"].max() < TRAIN_CUTOFF, "training window leaked into the holdout"

    results = []
    for k in K_GRID:
        for hfa in HFA_GRID:
            d_min, d_max, scale = fit_draw_model(train, k=k, hfa=hfa)
            preds, _ratings = run_elo(
                train, k=k, hfa=hfa, draw_min=d_min, draw_max=d_max, draw_scale=scale
            )
            ll = log_loss(preds[["p_home", "p_draw", "p_away"]].to_numpy(), _outcomes(preds))
            results.append((k, hfa, d_min, d_max, scale, ll))

    results.sort(key=lambda r: r[5])
    best_k, best_hfa, best_d_min, best_d_max, best_scale, best_ll = results[0]

    print(f"training window: {train['date'].min().date()} -> {train['date'].max().date()}  (n={len(train)})")
    print(f"{'K':>5} {'HFA':>5} {'log_loss':>10}")
    for k, hfa, _d_min, _d_max, _scale, ll in results:
        marker = "  <-- best" if (k, hfa) == (best_k, best_hfa) else ""
        print(f"{k:5d} {hfa:5d} {ll:10.4f}{marker}")

    print(f"\nbest: K={best_k}, HFA={best_hfa}, log_loss={best_ll:.4f}")
    print(f"matching draw curve: d_min={best_d_min:.6f}, d_max={best_d_max:.6f}, scale={best_scale:.4f}")


if __name__ == "__main__":
    main()