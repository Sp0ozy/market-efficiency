"""
tests/test_invariants.py — the assertion suite.

Written before models/elo.py exists (CLAUDE.md hard rule #4: assertions
before features). Two groups of tests:

  GREEN now -- chronology, probability normalization, and the scoring
  ground-truth anchors. These only need tests/fixtures.py and core/scoring.py,
  both of which already exist.

  RED by design -- four invariants that depend on a real Elo model and market
  adapter that haven't been built yet. Each imports its not-yet-existing
  module locally, inside the test body, so it fails with a clean
  ModuleNotFoundError instead of blocking collection of the green tests above.
  DO NOT weaken these thresholds to make them pass early -- the redness is
  the correct, expected signal that the model isn't built yet.
"""

import numpy as np
import pytest

from core.scoring import brier, log_loss
from tests.fixtures import fake_market_table, fake_model_table, to_arrays

# ---------------------------------------------------------------------------
# GREEN now
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
# RED by design -- until the real Elo model and market adapter exist
# ---------------------------------------------------------------------------


def test_model_correlates_with_market():
    """Elo's home-win probability must correlate with the market's (> 0.6).

    A correlation below 0.6 would mean home/away are swapped, team names are
    ghosted, or ratings never update -- so this stays a hard assertion, not
    a threshold to relax.
    """
    from models.elo import fit_predict  # red until the model exists

    p_elo = fit_predict()
    p_mkt = fake_market_table()
    assert np.corrcoef(p_elo[:, 0], p_mkt[:, 0])[0, 1] > 0.6


def test_elo_beats_base_rates():
    """Elo must beat the dumb base-rate baseline on log loss.

    If this ever fails once both exist, the model is BROKEN, not "efficient"
    -- do not reclassify a failure here as an acceptable outcome.
    """
    from core.baselines import base_rate_predictions  # red until this exists
    from models.elo import fit_predict  # red until the model exists

    p_elo = fit_predict()
    outcomes = fake_market_table()["outcome"]
    p_base = base_rate_predictions(outcomes)
    assert log_loss(p_elo, outcomes) < log_loss(p_base, outcomes)


def test_market_beats_elo():
    """THE CENTRAL ASSERTION. We assert that we LOSE to Pinnacle's close.

    If this ever fails once both are real: LOOKAHEAD BUG. Do not celebrate.
    Debug it.
    """
    from models.elo import fit_predict  # red until the model exists

    p_elo = fit_predict()
    market = fake_market_table()  # placeholder for the REAL Pinnacle-close table
    p_mkt, outcomes = to_arrays(market)
    assert log_loss(p_elo, outcomes) > log_loss(p_mkt, outcomes)


def test_no_ghost_teams():
    """Every team's rating must have moved away from the 1500 seed.

    A rating still at 1500.0 means that team never had a game processed --
    almost always a team-name mismatch, not a legitimately "average" team.
    """
    from models.elo import fit_ratings  # red until the model exists

    ratings = fit_ratings()
    assert (np.array(list(ratings.values())) != 1500.0).all()
