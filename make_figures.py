from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import cast

from adapters.football import load_matches
from core.scoring import log_loss
from holdout_report import build_holdout_table
from models.elo import rating_history

FIGURES = Path("reports/figures")

BLUE = "#2a78d6"       # Elo
RED = "#e34948"        # Pinnacle
GREY = "#898781"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
DIAGONAL = "#999999"

BIG6 = ["Arsenal", "Chelsea", "Liverpool", "Man City", "Man United", "Tottenham"]
BIG6_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": "#e1e0d9",
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 1.0,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})


def _clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e1e0d9")
    ax.spines["bottom"].set_color("#e1e0d9")
    ax.grid(axis="x", visible=False)


def _break_gaps(sub: pd.DataFrame, threshold_days: int = 120) -> pd.DataFrame:
    """Insert a NaN breaker row wherever a team's next match is far in the
    future (relegation-then-return), so the line shows a gap instead of a
    straight diagonal across seasons that team didn't play."""
    sub = sub.sort_values("date").reset_index(drop=True)
    rows = [sub.iloc[0]]
    for i in range(1, len(sub)):
        d_i = cast(pd.Timestamp, sub.loc[i, "date"])
        d_prev = cast(pd.Timestamp, sub.loc[i - 1, "date"])
        if (d_i - d_prev).days > threshold_days:
            mid = d_prev + (d_i - d_prev) / 2
            rows.append(pd.Series({"date": mid, "team": sub.loc[i, "team"], "rating": np.nan}))
        rows.append(sub.iloc[i])
    return pd.DataFrame(rows).reset_index(drop=True)


def fig_elo_trajectories(hist: pd.DataFrame) -> None:
    """All 42 teams' Elo ratings, full history. Big 6 highlighted."""
    fig, ax = plt.subplots(figsize=(11, 7))

    for team, sub in hist.groupby("team"):
        if team in BIG6:
            continue
        sub = _break_gaps(sub)
        ax.plot(sub["date"], sub["rating"], color=GREY, alpha=0.35, linewidth=0.8, zorder=1)

    for team, color in zip(BIG6, BIG6_COLORS):
        sub = _break_gaps(hist[hist["team"] == team])
        ax.plot(sub["date"], sub["rating"], color=color, linewidth=2, zorder=3)
        last = sub.iloc[-1]
        ax.plot(last["date"], last["rating"], marker="o", markersize=5, color=color, zorder=4)
        ax.annotate(team, (last["date"], last["rating"]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=10, color=INK_SECONDARY)

    ax.axhline(1500, linestyle="--", linewidth=1.0, color=DIAGONAL, zorder=1)
    ax.set_xlim(hist["date"].min(), hist["date"].max() + pd.Timedelta(days=250))
    ax.set_xlabel("date")
    ax.set_ylabel("Elo rating")
    ax.set_title("All 42 teams' Elo -- Big 6 highlighted", color=INK, fontsize=13, loc="left")
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "elo_trajectories.png", dpi=150)
    plt.close(fig)


def fig_season_trend(elo_h: pd.DataFrame, pinnacle_h: pd.DataFrame, y: np.ndarray) -> None:
    """Elo vs Pinnacle-close log loss by season, holdout only."""
    seasons = sorted(elo_h["season"].unique())
    elo_ll, mkt_ll = [], []
    for s in seasons:
        mask = (elo_h["season"] == s).to_numpy()
        p_elo = elo_h.loc[mask, ["p_home", "p_draw", "p_away"]].to_numpy()
        p_mkt = pinnacle_h.loc[mask, ["p_home", "p_draw", "p_away"]].to_numpy()
        elo_ll.append(log_loss(p_elo, y[mask]))
        mkt_ll.append(log_loss(p_mkt, y[mask]))

    labels = [f"{s[:2]}/{s[2:]}" for s in seasons]
    x = np.arange(len(seasons))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, elo_ll, marker="o", markersize=8, linewidth=2, color=BLUE, label="Elo")
    ax.plot(x, mkt_ll, marker="o", markersize=8, linewidth=2, color=RED, label="Pinnacle close")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("season")
    ax.set_ylabel("log loss (lower is better)")
    ax.set_title("Log loss by season, holdout only", color=INK, fontsize=13, loc="left")
    ax.legend(frameon=False, loc="upper left")
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "season_trend.png", dpi=150)
    plt.close(fig)


def fig_disagreement_bucket(elo_h: pd.DataFrame, pinnacle_h: pd.DataFrame, y: np.ndarray) -> None:
    """When Elo and the market pick different favorites, whose pick is
    actually right more often?"""
    p_elo = elo_h[["p_home", "p_draw", "p_away"]].to_numpy()
    p_mkt = pinnacle_h[["p_home", "p_draw", "p_away"]].to_numpy()
    elo_pick = p_elo.argmax(axis=1)
    mkt_pick = p_mkt.argmax(axis=1)
    disagree = elo_pick != mkt_pick

    n = int(disagree.sum())
    elo_acc = (elo_pick[disagree] == y[disagree]).mean()
    mkt_acc = (mkt_pick[disagree] == y[disagree]).mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    x = np.arange(2)
    ax.bar(x, [elo_acc, mkt_acc], width=0.5, color=[BLUE, RED])
    for xi, v in zip(x, [elo_acc, mkt_acc]):
        ax.annotate(f"{v:.1%}", (xi, v), xytext=(0, 8), textcoords="offset points",
                    ha="center", color=INK, fontsize=13, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["Elo", "Pinnacle close"])
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("pick accuracy")
    ax.set_title(f"When Elo and market disagree (n={n}), trust the market",
                 color=INK, fontsize=12.5, loc="left")
    _clean_axes(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "disagreement_bucket.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    hist = rating_history(load_matches())
    elo_h, aligned_sources, y, _holdout_start, _train_outcomes = build_holdout_table()
    pinnacle_h = aligned_sources["pinnacle_close"]

    fig_elo_trajectories(hist)
    fig_season_trend(elo_h, pinnacle_h, y)
    fig_disagreement_bucket(elo_h, pinnacle_h, y)

    print(f"wrote 3 figures to {FIGURES}/")


if __name__ == "__main__":
    main()