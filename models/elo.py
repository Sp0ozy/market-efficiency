import pandas as pd

DEFAULT_K = 20.0    # a starting point; tuned later
DEFAULT_HFA = 100.0
DRAW_RATE = 0.25    # fixed, not tuned -- close to E0's long-run draw rate

_SCORE = {"H": 1.0, "D": 0.5, "A": 0.0}


def run_elo(matches: pd.DataFrame, k: float = DEFAULT_K, hfa: float = DEFAULT_HFA) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run Elo on a fixture list, returning predictions and final ratings."""
    
    ratings: dict[str, float] = {}
    rows = []

    for row in matches.itertuples(index=False):
        r_home = ratings.get(str(row.home), 1500.0)
        r_away = ratings.get(str(row.away), 1500.0)

        diff = (r_home + hfa) - r_away
        e_home = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

        p_home = (1.0 - DRAW_RATE) * e_home
        p_away = (1.0 - DRAW_RATE) * (1.0 - e_home)
        p_draw = DRAW_RATE

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
