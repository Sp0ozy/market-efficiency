import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Tuned via tune.py grid search on the training window only (date < 2019-01-01,
DEFAULT_K = 25.0
DEFAULT_HFA = 75.0
DRAW_MIN = 0.041548
DRAW_MAX = 0.294611
DRAW_SCALE = 285.9921

_SCORE = {"H": 1.0, "D": 0.5, "A": 0.0}


def draw_rate(diff: float, d_min: float, d_max: float, scale: float) -> float:
    return d_min + (d_max - d_min) * np.exp(-(diff / scale) ** 2)

def run_elo(
    matches: pd.DataFrame,
    k: float = DEFAULT_K,
    hfa: float = DEFAULT_HFA,
    draw_min: float = DRAW_MIN,
    draw_max: float = DRAW_MAX,
    draw_scale: float = DRAW_SCALE,
) -> tuple[pd.DataFrame, dict[str, float]]: 
    
    """Run Elo on a fixture list, returning predictions and final ratings."""
    
    ratings: dict[str, float] = {}
    rows = []

    for row in matches.itertuples(index=False):
        r_home = ratings.get(str(row.home), 1500.0)
        r_away = ratings.get(str(row.away), 1500.0)

        diff = (r_home + hfa) - r_away
        e_home = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

        p_draw = draw_rate(diff, draw_min, draw_max, draw_scale)
        p_home = (1.0 - p_draw) * e_home
        p_away = (1.0 - p_draw) * (1.0 - e_home)

        rows.append({
            "date": row.date, "div": row.div, "season": row.season,
            "home": row.home, "away": row.away,
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "outcome": row.outcome,
        })

        s_home = _SCORE[str(row.outcome)]
        ratings[str(row.home)] = r_home + k * (s_home - e_home)
        ratings[str(row.away)] = r_away + k * ((1.0 - s_home) - (1.0 - e_home))

    predictions = pd.DataFrame(rows)
    return predictions, ratings

def fit_draw_model(
    matches: pd.DataFrame, k: float = DEFAULT_K, hfa: float = DEFAULT_HFA
) -> tuple[float, float, float]:
    """
    Returns (d_min, d_max, scale) for d_min + (d_max - d_min) * exp(-(diff/scale)**2).
    """
    ratings: dict[str, float] = {}
    diffs, is_draw = [], []

    for row in matches.itertuples(index=False):
        r_home = ratings.get(str(row.home), 1500.0)
        r_away = ratings.get(str(row.away), 1500.0)
        diff = (r_home + hfa) - r_away
        diffs.append(diff)
        is_draw.append(1.0 if row.outcome == "D" else 0.0)

        e_home = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        s_home = _SCORE[str(row.outcome)]
        ratings[str(row.home)] = r_home + k * (s_home - e_home)
        ratings[str(row.away)] = r_away + k * ((1.0 - s_home) - (1.0 - e_home))

    diffs = np.array(diffs)
    is_draw = np.array(is_draw)

    edges = np.linspace(diffs.min(), diffs.max(), 21)
    bucket_x, bucket_y = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (diffs >= lo) & (diffs < hi)
        if mask.sum() >= 20:
            bucket_x.append(diffs[mask].mean())
            bucket_y.append(is_draw[mask].mean())

    (d_min, d_max, scale), _ = curve_fit(
        draw_rate, bucket_x, bucket_y, p0=[0.15, 0.30, 200.0], bounds=([0, 0, 10], [0.5, 0.6, 1000])
    )
    
    return float(d_min), float(d_max), float(scale)