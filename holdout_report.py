"""
holdout_report.py — touch the holdout. Once.

The holdout starts at the first match of season 1920 and is scored here
against four market sources (Pinnacle close/early, Bet365 close/early),
after K/HFA/draw curve were tuned on the training window only (see tune.py).
This script is that one touch: the headline log-loss table, the
favorite-longshot bias test, and the Kelly bankroll simulation.

Run: python holdout_report.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from adapters.football import (
    load_matches,
    season_start,
    pinnacle_close,
    pinnacle_early,
    bet365_close,
    bet365_early,
    pinnacle_close_odds,
)
from core.baselines import home_always_predictions, uniform_predictions
from core.bias import bias_table
from core.scoring import brier, log_loss
from core.staking import bankroll_simulation, best_edge_bets
from models.elo import run_elo

REPORTS = Path("reports")
_OUTCOME_TO_INT = {"H": 0, "D": 1, "A": 2}
_KEYS = ["date", "home", "away"]


def _outcomes(df: pd.DataFrame) -> np.ndarray:
    return df["outcome"].map(_OUTCOME_TO_INT).to_numpy()


def build_holdout_table():
    """One row per holdout match, aligned across Elo and all four market sources.

    Elo is trained walk-forward over the full history; only the scoring is
    restricted to the holdout window. The window is the intersection of all
    four market sources' coverage, so every model is compared on exactly the
    same matches.
    """
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
    odds_close = pinnacle_close_odds()

    shared = sources["pinnacle_close"][_KEYS]
    for name in ("pinnacle_early", "bet365_close", "bet365_early"):
        shared = shared.merge(sources[name][_KEYS], on=_KEYS, how="inner")
    shared = shared.merge(odds_close[_KEYS], on=_KEYS, how="inner")
    shared = shared[shared["date"] >= holdout_start]

    def _aligned(table: pd.DataFrame) -> pd.DataFrame:
        return (
            table.merge(shared, on=_KEYS, how="inner")
            .sort_values("date", kind="stable")
            .reset_index(drop=True)
        )

    elo_h = _aligned(elo_preds)
    aligned_sources = {name: _aligned(t) for name, t in sources.items()}
    odds_h = _aligned(odds_close)

    y = _outcomes(elo_h)
    for name, t in aligned_sources.items():
        assert (y == _outcomes(t)).all(), f"{name} outcomes don't line up with elo"
    assert (y == _outcomes(odds_h)).all(), "odds outcomes don't line up with elo"

    return elo_h, aligned_sources, odds_h, y, holdout_start, train_outcomes


def report_scores(
    elo_h: pd.DataFrame,
    aligned_sources: dict[str, pd.DataFrame],
    y: np.ndarray,
    train_outcomes: np.ndarray,
) -> pd.DataFrame:
    """Headline log-loss/brier table, holdout only. base_rate uses the
    TRAINING window's outcome frequency, not the holdout's own (using the
    holdout's own realized rate would be leakage)."""
    n = len(y)
    train_rate = np.bincount(train_outcomes, minlength=3) / len(train_outcomes)

    predictions = {
        "uniform": uniform_predictions(n),
        "base_rate (train-only rate)": np.tile(train_rate, (n, 1)),
        "home_always": home_always_predictions(n),
        "elo": elo_h[["p_home", "p_draw", "p_away"]].to_numpy(),
    }
    for name, t in aligned_sources.items():
        predictions[name] = t[["p_home", "p_draw", "p_away"]].to_numpy()

    report = pd.DataFrame(
        {name: {"log_loss": log_loss(p, y), "brier": brier(p, y)} for name, p in predictions.items()}
    ).T
    report.index.name = "model"
    return report


def report_bias(elo_h: pd.DataFrame, pinnacle_h: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Favorite-longshot bias test, Elo and Pinnacle close side by side."""
    p_elo = elo_h[["p_home", "p_draw", "p_away"]].to_numpy()
    p_mkt = pinnacle_h[["p_home", "p_draw", "p_away"]].to_numpy()

    mean_pred_elo, freq_elo, ci_lo_elo, ci_hi_elo, resid_elo, sig_elo, count_elo = bias_table(p_elo, y)
    mean_pred_mkt, freq_mkt, ci_lo_mkt, ci_hi_mkt, resid_mkt, sig_mkt, count_mkt = bias_table(p_mkt, y)

    n = min(len(mean_pred_elo), len(mean_pred_mkt))
    return pd.DataFrame({
        "mean_predicted_elo": mean_pred_elo[:n], "empirical_freq_elo": freq_elo[:n],
        "ci_lo_elo": ci_lo_elo[:n], "ci_hi_elo": ci_hi_elo[:n],
        "residual_elo": resid_elo[:n], "significant_elo": sig_elo[:n], "count_elo": count_elo[:n],
        "mean_predicted_mkt": mean_pred_mkt[:n], "empirical_freq_mkt": freq_mkt[:n],
        "ci_lo_mkt": ci_lo_mkt[:n], "ci_hi_mkt": ci_hi_mkt[:n],
        "residual_mkt": resid_mkt[:n], "significant_mkt": sig_mkt[:n], "count_mkt": count_mkt[:n],
    })


def report_staking(elo_h: pd.DataFrame, odds_h: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Kelly-sized bankroll sim: bet Elo's edge against Pinnacle's real quoted
    (vig-included) odds. Full Kelly and half Kelly, both from bankroll=1.0."""
    p_elo = elo_h[["p_home", "p_draw", "p_away"]].to_numpy()
    odds = odds_h[["odds_home", "odds_draw", "odds_away"]].to_numpy()

    chosen_col, stake_fraction, _edge = best_edge_bets(p_elo, odds)
    safe_col = np.where(chosen_col == -1, 0, chosen_col)
    odds_chosen = odds[np.arange(len(y)), safe_col]
    hit = (chosen_col == y) & (chosen_col != -1)

    mask = stake_fraction > 0
    win_rate = hit[mask].mean()
    breakeven = (1.0 / odds_chosen[mask]).mean()
    flat_roi = (hit[mask] * (odds_chosen[mask] - 1.0) - (~hit[mask])).mean()
    print(f"bets placed: {int(mask.sum())} / {len(y)} holdout matches")
    print(f"avg quoted odds on chosen bets: {odds_chosen[mask].mean():.2f}  "
          f"(flat-stake breakeven win rate {breakeven:.1%}, actual {win_rate:.1%}, "
          f"flat-stake ROI {flat_roi:+.1%})")
    print("Kelly sizing assumes the input probabilities ARE the true probabilities;")
    print("Elo's are miscalibrated relative to reality, so even a near-flat realized")
    print("edge compounds into ruin once you size bets as if that edge were real.")

    full = bankroll_simulation(stake_fraction, odds_chosen, hit, kelly_multiplier=1.0)
    half = bankroll_simulation(stake_fraction, odds_chosen, hit, kelly_multiplier=0.5)

    print(f"\nfull Kelly:  final bankroll {full[-1]:.6f}  (start 1.0)  max drawdown {1 - (full / np.maximum.accumulate(full)).min():.2%}")
    print(f"half Kelly:  final bankroll {half[-1]:.6f}  (start 1.0)  max drawdown {1 - (half / np.maximum.accumulate(half)).min():.2%}")
    print("ONE equity curve is one sample path, not evidence.")

    return pd.DataFrame({"step": np.arange(len(full)), "full_kelly": full, "half_kelly": half})


def main() -> None:
    elo_h, aligned_sources, odds_h, y, holdout_start, train_outcomes = build_holdout_table()
    n = len(y)
    print(f"holdout start (season 1920): {holdout_start.date()}")
    print(f"holdout window (shared across all 4 market sources): "
          f"{elo_h['date'].min().date()} -> {elo_h['date'].max().date()}  (n={n})\n")

    REPORTS.mkdir(exist_ok=True)

    print("--- log loss / brier (holdout only) ---")
    scores = report_scores(elo_h, aligned_sources, y, train_outcomes)
    scores.to_csv(REPORTS / "holdout_scores.csv")
    print(scores.round(4).to_string())
    if scores.loc["elo", "log_loss"] < scores.loc["pinnacle_close", "log_loss"]:
        print("\n*** Elo beat the market on the holdout. This is a LOOKAHEAD BUG,")
        print("*** not a discovery. Do not celebrate -- debug it.")

    for name in ("pinnacle_close", "bet365_close"):
        corr = np.corrcoef(elo_h["p_home"], aligned_sources[name]["p_home"])[0, 1]
        print(f"corr(p_home elo, p_home {name}) = {corr:.3f}")

    print("\n--- favorite-longshot bias (Elo vs Pinnacle close, holdout only) ---")
    bias = report_bias(elo_h, aligned_sources["pinnacle_close"], y)
    bias.to_csv(REPORTS / "bias_table.csv", index=False)
    print(bias.round(4).to_string(index=False))
    corr = np.corrcoef(bias["mean_predicted_mkt"], bias["residual_mkt"])[0, 1]
    verdict = "FOUND" if (corr > 0 and bias["significant_mkt"].any()) else "NOT CONCLUSIVELY FOUND"
    print(f"\ncorr(mean_predicted, residual), Pinnacle close = {corr:.3f}  ->  favorite-longshot bias: {verdict}")
    print("(multiplicative de-vig inflates longshot probabilities -- this can manufacture")
    print(" an artifact shaped exactly like favorite-longshot bias. State this limitation.)")

    print("\n--- Kelly staking, vig included (Elo edge vs Pinnacle close) ---")
    bankroll = report_staking(elo_h, odds_h, y)
    bankroll.to_csv(REPORTS / "bankroll.csv", index=False)


if __name__ == "__main__":
    main()
