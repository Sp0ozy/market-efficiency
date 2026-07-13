"""
tests/test_invariants.py — the assertion suite.

Per CLAUDE.md hard rule #4 ("assertions before features") this suite was
started before models/elo.py existed, with four invariants deliberately red.
Now that adapters/football.py, core/baselines.py, and models/elo.py all
exist, those four run against the REAL pipeline (full match history,
Pinnacle-closing eval window) instead of random fixtures -- fixtures can't
prove correlation or ordering claims since they're independently random by
construction.

  GREEN, fixture-based -- chronology, probability normalization, and the
  scoring ground-truth anchors. These only need tests/fixtures.py and
  core/scoring.py.

  GREEN, real-pipeline-based -- the four invariants. Each still imports its
  pipeline modules lazily (via a fixture), so a regression in any one module
  fails only the tests that touch it. DO NOT weaken these thresholds to make
  a future regression pass -- a failure here means a lookahead bug or a
  broken rating loop, never "the market is more efficient than we thought".
"""

import numpy as np
import pytest

from core.scoring import brier, log_loss
from tests.fixtures import fake_market_table, fake_model_table, to_arrays

# ---------------------------------------------------------------------------
# GREEN, fixture-based
# ---------------------------------------------------------------------------


def test_chronology():
    """Fixture tables must be sorted by date, ascending."""
    market = fake_market_table()
    model = fake_model_table()
    assert market["date"].is_monotonic_increasing
    assert model["date"].is_monotonic_increasing


def test_probabilities_normalized():
    """Every prediction row (fake market and fake model) sums to 1.0."""
    for table in (fake_market_table(), fake_model_table()):
        probs, _ = to_arrays(table)
        assert np.allclose(probs.sum(axis=1), 1.0)


def test_log_loss_uniform_ground_truth():
    """log_loss under uniform (1/3, 1/3, 1/3) predictions must equal ln(3)."""
    probs = np.full((4, 3), 1 / 3)
    outcomes = np.array([0, 1, 2, 0])
    assert log_loss(probs, outcomes) == pytest.approx(1.0986122886681098)


def test_brier_uniform_ground_truth():
    """brier under uniform (1/3, 1/3, 1/3) predictions must equal 2/3."""
    probs = np.full((4, 3), 1 / 3)
    outcomes = np.array([0, 1, 2, 0])
    assert brier(probs, outcomes) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# GREEN, real-pipeline-based
# ---------------------------------------------------------------------------

_OUTCOME_TO_INT = {"H": 0, "D": 1, "A": 2}


@pytest.fixture(scope="module")
def real_pipeline():
    """Walk-forward Elo over the FULL match history, plus the market table.

    Loaded lazily (only when a test requests this fixture) so a missing
    pipeline module fails just the tests that depend on it, not collection
    of the fixture-based tests above.
    """
    from adapters.football import load_matches, market_table
    from models.elo import run_elo

    matches = load_matches()
    elo_preds, ratings = run_elo(matches)
    market = market_table()

    # Elo trains on all matches, but is only compared to the market over the
    # Pinnacle-closing eval window -- the market's own coverage defines it.
    eval_elo = elo_preds.merge(
        market[["date", "home", "away"]], on=["date", "home", "away"], how="inner"
    )
    return elo_preds, ratings, market, eval_elo


def test_model_correlates_with_market(real_pipeline):
    """Elo's home-win probability must correlate with the market's (> 0.6)."""
    _elo_preds, _ratings, market, eval_elo = real_pipeline
    corr = np.corrcoef(eval_elo["p_home"], market["p_home"])[0, 1]
    assert corr > 0.6


def test_elo_beats_base_rates(real_pipeline):
    """Elo must beat the dumb base-rate baseline on log loss, over its full history."""
    from core.baselines import base_rate_predictions

    elo_preds, _ratings, _market, _eval_elo = real_pipeline
    outcomes = elo_preds["outcome"].map(_OUTCOME_TO_INT).to_numpy()
    p_elo = elo_preds[["p_home", "p_draw", "p_away"]].to_numpy()
    p_base = base_rate_predictions(outcomes)
    assert log_loss(p_elo, outcomes) < log_loss(p_base, outcomes)


def test_market_beats_elo(real_pipeline):
    """THE CENTRAL ASSERTION. We assert that we LOSE to Pinnacle's close."""
    _elo_preds, _ratings, market, eval_elo = real_pipeline

    outcomes = eval_elo["outcome"].map(_OUTCOME_TO_INT).to_numpy()
    outcomes_mkt = market["outcome"].map(_OUTCOME_TO_INT).to_numpy()
    assert (outcomes == outcomes_mkt).all()

    p_elo = eval_elo[["p_home", "p_draw", "p_away"]].to_numpy()
    p_mkt = market[["p_home", "p_draw", "p_away"]].to_numpy()
    assert log_loss(p_elo, outcomes) > log_loss(p_mkt, outcomes_mkt)


def test_no_ghost_teams(real_pipeline):
    """Every team's rating must have moved away from the 1500 seed."""
    _elo_preds, ratings, _market, _eval_elo = real_pipeline
    assert (np.array(list(ratings.values())) != 1500.0).all()