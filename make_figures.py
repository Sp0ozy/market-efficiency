from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import cast

from adapters.football import load_matches
from core.scoring import log_loss
from holdout_report import build_holdout_table
from models.elo import rating_history
from core.bias import bias_table
from explore_segments import confusion_matrix, biggest_disagreements, team_disagreement_frequency

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


def fig_bias(elo_h: pd.DataFrame, pinnacle_h: pd.DataFrame, y: np.ndarray) -> None:
    """Favorite-longshot bias: Elo vs Pinnacle close, calibration residual
    with Wilson-score CIs, overlaid. Opposite-signed correlations here are
    evidence the market's pattern reflects genuine pricing, not a shared
    data/devig artifact -- see explore_segments.py's bias_table output."""
    p_elo = elo_h[["p_home", "p_draw", "p_away"]].to_numpy()
    p_mkt = pinnacle_h[["p_home", "p_draw", "p_away"]].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color=DIAGONAL, zorder=1)

    for label, p, color in [("Elo", p_elo, BLUE), ("Pinnacle close", p_mkt, RED)]:
        mean_pred, emp, ci_lo, ci_hi, _resid, _sig, _counts = bias_table(p, y)
        yerr = np.vstack([
            np.clip(emp - ci_lo, 0, None),
            np.clip(ci_hi - emp, 0, None),
        ])
        ax.errorbar(mean_pred, emp, yerr=yerr, fmt="o", markersize=7, color=color,
                    ecolor=color, elinewidth=1.3, capsize=3, label=label, zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("empirical frequency (95% Wilson CI)")
    ax.set_title("Favorite-longshot bias: Elo vs Pinnacle close", color=INK, fontsize=13, loc="left")
    ax.legend(frameon=False, loc="upper left")
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "bias.png", dpi=150)
    plt.close(fig)

def fig_confusion_matrix(cm: pd.DataFrame) -> None:
    """Heatmap of Elo's pick x market's pick. The empty draw row/column
    isn't a bug -- neither model's top pick is ever a draw (see
    explore_segments.py's confusion_matrix for why: mathematically
    guaranteed here since DRAW_MAX < 1/3)."""
    fig, ax = plt.subplots(figsize=(7, 6.5))
    data = cm.to_numpy()
    ax.imshow(data, cmap="Blues", vmin=0)

    ax.set_xticks(range(3))
    ax.set_xticklabels(cm.columns)
    ax.set_yticks(range(3))
    ax.set_yticklabels(cm.index)

    for i in range(3):
        for j in range(3):
            v = data[i, j]
            text_color = "white" if v > data.max() * 0.5 else INK
            ax.text(j, i, f"{v:,}", ha="center", va="center", color=text_color, fontsize=13, fontweight="bold")

    ax.set_xlabel("market's pick")
    ax.set_ylabel("Elo's pick")
    ax.set_title("Elo's pick vs market's pick", color=INK, fontsize=13, loc="left")
    ax.text(0.0, -0.18, "Draw row/column empty by construction -- neither model's top pick is ever a draw.",
            transform=ax.transAxes, fontsize=9, color=INK_SECONDARY)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=2)
    ax.tick_params(which="minor", size=0)
    ax.grid(which="major", visible=False)

    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    
    
def fig_disagreement_scatter(elo_h: pd.DataFrame, pinnacle_h: pd.DataFrame, bd: pd.DataFrame) -> None:
    """Every holdout match's Elo vs Pinnacle home-win probability. Points
    near the diagonal agree; the biggest few disagreements (from
    explore_segments.py's biggest_disagreements table) are labeled."""
    p_elo_home = elo_h["p_home"].to_numpy()
    p_mkt_home = pinnacle_h["p_home"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color=DIAGONAL, zorder=1)
    ax.scatter(p_mkt_home, p_elo_home, s=14, color=GREY, alpha=0.35, zorder=2)

    top5 = bd.head(5)
    for _, row in top5.iterrows():
        ax.scatter(row["mkt_p_home"], row["elo_p_home"], s=50, color=RED, zorder=3)
        ax.annotate(f"{row['home']} vs {row['away']}", (row["mkt_p_home"], row["elo_p_home"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK_SECONDARY)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("market's p(home win)")
    ax.set_ylabel("Elo's p(home win)")
    ax.set_title("Every match: Elo vs market -- biggest disagreements", color=INK, fontsize=13, loc="left")
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "disagreement_scatter.png", dpi=150)
    plt.close(fig)
    
def fig_team_disagreement(td: pd.DataFrame, n: int = 10) -> None:
    """Top n teams by disagreement rate (>=20 matches only -- see
    explore_segments.py's reliability flag)."""
    reliable = td[td["reliable"]].head(n).iloc[::-1]  # reverse so #1 is on top

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(reliable["team"], reliable["disagree_rate"], color=BLUE)
    for team, rate in zip(reliable["team"], reliable["disagree_rate"]):
        ax.annotate(f"{rate:.1%}", (rate, team), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=10)

    ax.set_xlabel("disagreement rate")
    ax.set_title(f"Top {n} teams by Elo/market disagreement rate", color=INK, fontsize=13, loc="left")
    _clean_axes(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()  
    fig.savefig(FIGURES / "team_disagreement.png", dpi=150)
    plt.close(fig)
    
def fig_extreme_summary(elo_h: pd.DataFrame, pinnacle_h: pd.DataFrame, y: np.ndarray, n: int = 20) -> None:
    """Elo's confidence vs market's price on the same pick vs actual rate,
    for the n most lopsided matches by Elo's own rating gap."""
    p_elo = elo_h[["p_home", "p_draw", "p_away"]].to_numpy()
    p_mkt = pinnacle_h[["p_home", "p_draw", "p_away"]].to_numpy()
    elo_conf = p_elo.max(axis=1)
    elo_fav = p_elo.argmax(axis=1)
    order = np.argsort(-elo_conf)[:n]
    fav_idx = elo_fav[order]

    values = [elo_conf[order].mean(), p_mkt[order, fav_idx].mean(), (fav_idx == y[order]).mean()]
    labels = ["Elo's\nconfidence", "Market's price\non same pick", "Actual\nwin rate"]
    colors = [BLUE, RED, GREY]

    fig, ax = plt.subplots(figsize=(7, 6))
    x = np.arange(3)
    ax.bar(x, values, width=0.5, color=colors)
    for xi, v in zip(x, values):
        ax.annotate(f"{v:.1%}", (xi, v), xytext=(0, 8), textcoords="offset points",
                    ha="center", color=INK, fontsize=13, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Top {n} most lopsided matches by Elo's rating gap", color=INK, fontsize=12.5, loc="left")
    _clean_axes(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "extreme_summary.png", dpi=150)
    plt.close(fig)



def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    hist = rating_history(load_matches())
    elo_h, aligned_sources, y, _holdout_start, _train_outcomes = build_holdout_table()
    pinnacle_h = aligned_sources["pinnacle_close"]

    fig_elo_trajectories(hist)
    fig_season_trend(elo_h, pinnacle_h, y)
    fig_disagreement_bucket(elo_h, pinnacle_h, y)
    fig_bias(elo_h, pinnacle_h, y)

    cm = confusion_matrix(elo_h, pinnacle_h, y)
    fig_confusion_matrix(cm)

    bd = biggest_disagreements(elo_h, pinnacle_h, y, n=20)
    fig_disagreement_scatter(elo_h, pinnacle_h, bd)

    td = team_disagreement_frequency(elo_h, pinnacle_h, y)
    fig_team_disagreement(td)

    fig_extreme_summary(elo_h, pinnacle_h, y)

    print(f"wrote figures to {FIGURES}/")


if __name__ == "__main__":
    main()