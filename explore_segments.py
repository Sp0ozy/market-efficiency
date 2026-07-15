"""
explore_segments.py — EXPLORATORY characterization of where and why Elo
disagrees with the market. NOT a new confirmatory finding.

CLAUDE.md hard rule #3: the holdout is touched once, at the end. This script
touches it again, deliberately, slicing it several ways to characterize
disagreement. With ~2,000+ holdout matches sliced multiple ways, some
pattern will look interesting by chance -- treat everything here as
hypothesis-generating for the presentation, not a new statistical claim.

Run: python explore_segments.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from holdout_report import build_holdout_table

OUT = Path("reports/exploratory")
BANNER = "(exploratory -- characterizes disagreement, not a new confirmatory finding)"


def confusion_matrix(elo_h: pd.DataFrame, pinnacle_h: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Elo's pick x market's pick: 3x3 counts."""
    p_elo = elo_h[["p_home", "p_draw", "p_away"]].to_numpy()
    p_mkt = pinnacle_h[["p_home", "p_draw", "p_away"]].to_numpy()
    elo_pick = p_elo.argmax(axis=1)
    mkt_pick = p_mkt.argmax(axis=1)

    labels = ["H", "D", "A"]
    counts = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            counts[i, j] = int(((elo_pick == i) & (mkt_pick == j)).sum())

    return pd.DataFrame(counts, index=[f"elo:{l}" for l in labels], columns=[f"mkt:{l}" for l in labels])


def biggest_disagreements(elo_h: pd.DataFrame, pinnacle_h: pd.DataFrame, y: np.ndarray, n: int = 20) -> pd.DataFrame:
    """Top n matches by |p_home_elo - p_home_market|."""
    p_elo = elo_h[["p_home", "p_draw", "p_away"]].to_numpy()
    p_mkt = pinnacle_h[["p_home", "p_draw", "p_away"]].to_numpy()
    gap = np.abs(p_elo[:, 0] - p_mkt[:, 0])
    order = np.argsort(-gap)[:n]

    rows = []
    for i in order:
        rows.append({
            "date": elo_h.iloc[i]["date"], "home": elo_h.iloc[i]["home"], "away": elo_h.iloc[i]["away"],
            "elo_p_home": p_elo[i, 0], "mkt_p_home": p_mkt[i, 0], "gap": gap[i],
            "outcome": elo_h.iloc[i]["outcome"],
        })
    return pd.DataFrame(rows)


def team_disagreement_frequency(elo_h: pd.DataFrame, pinnacle_h: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """For each team, how often are they in a match where Elo and the market
    disagree, out of how many total matches. Teams with < 20 matches are
    flagged -- same suspicious-sample-size threshold as clean.py's NAME_MAP
    check, for the same reason: too few matches to say anything real."""
    p_elo = elo_h[["p_home", "p_draw", "p_away"]].to_numpy()
    p_mkt = pinnacle_h[["p_home", "p_draw", "p_away"]].to_numpy()
    elo_pick = p_elo.argmax(axis=1)
    mkt_pick = p_mkt.argmax(axis=1)
    disagree = elo_pick != mkt_pick

    teams = pd.concat([elo_h["home"], elo_h["away"]]).unique()
    rows = []
    for team in teams:
        mask = ((elo_h["home"] == team) | (elo_h["away"] == team)).to_numpy()
        n_matches = int(mask.sum())
        n_disagree = int(disagree[mask].sum())
        rows.append({
            "team": team, "n_matches": n_matches, "n_disagree": n_disagree,
            "disagree_rate": n_disagree / n_matches if n_matches else 0.0,
            "reliable": n_matches >= 20,
        })
    return pd.DataFrame(rows).sort_values("disagree_rate", ascending=False).reset_index(drop=True)


def extreme_matches_table(elo_h: pd.DataFrame, pinnacle_h: pd.DataFrame, y: np.ndarray, n: int = 20) -> pd.DataFrame:
    """The n matches where Elo's own rating gap implies the most lopsided
    result (its max probability across H/D/A) -- a concrete look at how
    calibration holds up where a "power rating" gap is largest."""
    p_elo = elo_h[["p_home", "p_draw", "p_away"]].to_numpy()
    p_mkt = pinnacle_h[["p_home", "p_draw", "p_away"]].to_numpy()
    elo_conf = p_elo.max(axis=1)
    elo_fav = p_elo.argmax(axis=1)

    order = np.argsort(-elo_conf)[:n]

    rows = []
    for i in order:
        rows.append({
            "date": elo_h.iloc[i]["date"], "home": elo_h.iloc[i]["home"], "away": elo_h.iloc[i]["away"],
            "elo_p_home": p_elo[i, 0], "elo_p_draw": p_elo[i, 1], "elo_p_away": p_elo[i, 2],
            "mkt_p_home": p_mkt[i, 0], "mkt_p_draw": p_mkt[i, 1], "mkt_p_away": p_mkt[i, 2],
            "outcome": elo_h.iloc[i]["outcome"],
            "elo_favorite_won": bool(elo_fav[i] == y[i]),
        })
    table = pd.DataFrame(rows)

    fav_idx = elo_fav[order]
    won = fav_idx == y[order]
    print(f"Top {n} most lopsided matches by Elo's own rating gap:")
    print(f"  Elo's average confidence in its favorite:       {elo_conf[order].mean():.1%}")
    print(f"  Market's average prob on that SAME outcome:     {p_mkt[order, fav_idx].mean():.1%}")
    print(f"  Actual rate that favorite won:                  {won.mean():.1%}")

    return table


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    elo_h, aligned_sources, y, _holdout_start, _train_outcomes = build_holdout_table()
    pinnacle_h = aligned_sources["pinnacle_close"]

    print(f"{'=' * 70}\nEXPLORATORY SEGMENT ANALYSIS\n{BANNER}\n{'=' * 70}")

    print("\n--- confusion matrix (rows=Elo's pick, cols=market's pick) ---")
    cm = confusion_matrix(elo_h, pinnacle_h, y)
    print(cm.to_string())
    cm.to_csv(OUT / "confusion_matrix.csv")

    print("\n--- biggest disagreements ---")
    bd = biggest_disagreements(elo_h, pinnacle_h, y, n=20)
    print(bd.round(3).to_string(index=False))
    bd.to_csv(OUT / "biggest_disagreements.csv", index=False)

    print("\n--- team disagreement frequency (>=20 matches only, shown) ---")
    td = team_disagreement_frequency(elo_h, pinnacle_h, y)
    print(td[td["reliable"]].head(10).to_string(index=False))
    td.to_csv(OUT / "team_disagreement.csv", index=False)

    print("\n--- extreme matches (biggest 'power rating' mismatches) ---")
    em = extreme_matches_table(elo_h, pinnacle_h, y, n=20)
    em.to_csv(OUT / "extreme_matches.csv", index=False)

    print(f"\n{BANNER}")


if __name__ == "__main__":
    main()
