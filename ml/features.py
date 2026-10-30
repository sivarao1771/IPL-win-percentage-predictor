"""Feature definitions for a T20 chase.

Data preparation, training and the API all describe a chase the same way, so the
definitions live here in one place.
"""
import numpy as np

TOTAL_BALLS = 120
TOTAL_WICKETS = 10

FEATURE_COLUMNS = ["runs_left", "balls_left", "wickets_left", "runs_target", "crr", "rrr"]
CATEGORY_COLUMNS = ["batting_team", "bowling_team", "city"]

# Match phases for reporting, by legal balls bowled at the moment of prediction.
# The pre-chase state (0 balls) belongs to the powerplay.
PHASES = [("overs 1-6", 0, 36), ("overs 7-15", 37, 90), ("overs 16-20", 91, 119)]


def compute_features(runs_target, team_runs, balls_bowled, wickets_lost):
    """Turn the raw state of a chase into model features.

    Works on plain numbers or on pandas Series of equal length. The state must be
    undecided (at least one ball left); otherwise rrr is NaN so validation catches it.
    """
    runs_left = runs_target - team_runs
    balls_left = TOTAL_BALLS - balls_bowled
    crr = np.where(balls_bowled > 0, team_runs * 6 / np.maximum(balls_bowled, 1), 0.0)
    rrr = np.where(balls_left > 0, runs_left * 6 / np.maximum(balls_left, 1), np.nan)
    return {
        "runs_left": runs_left,
        "balls_left": balls_left,
        "wickets_left": TOTAL_WICKETS - wickets_lost,
        "runs_target": runs_target,
        "crr": crr,
        "rrr": rrr,
    }


def phase_of(balls_bowled):
    """Name of the match phase for a number of legal balls bowled."""
    for name, low, high in PHASES:
        if low <= balls_bowled <= high:
            return name
    raise ValueError(f"balls_bowled out of range: {balls_bowled}")
