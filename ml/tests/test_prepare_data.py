import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features import compute_features, phase_of  # noqa: E402
from prepare_data import (  # noqa: E402
    MANUAL_EXCLUSIONS, SPLITS, build_states, exclusion_reason, kept_by_rule, match_outline,
    split_by_season, validate,
)

HOME, AWAY = "Mumbai Indians", "Chennai Super Kings"


def innings_rows(match_id, innings, batting, bowling, balls, runs_per_ball, wickets_at=(), target=None):
    """Ball-by-ball rows for one innings. Running totals mirror the real CSV columns."""
    rows, runs, wickets = [], 0, 0
    for ball in range(1, balls + 1):
        runs += runs_per_ball
        wickets += ball in wickets_at
        rows.append({
            "match_id": match_id, "date": "2015-04-10", "innings": innings,
            "batting_team": batting, "bowling_team": bowling, "city": "Mumbai",
            "over": (ball - 1) // 6, "player_out": "batter" if ball in wickets_at else None,
            "team_runs": runs, "team_balls": ball, "team_wicket": wickets,
            "runs_target": target, "match_won_by": None, "result_type": None, "method": None,
        })
    return rows


def make_match(match_id, winner, inn1_balls=120, inn1_wickets_at=(), chase_balls=60,
               chase_runs_per_ball=2, chase_wickets_at=(), result_type=None, method=None):
    """First innings at 1 run a ball (so target = inn1_balls + 1), then the chase."""
    target = inn1_balls + 1
    rows = (innings_rows(match_id, 1, HOME, AWAY, inn1_balls, 1, inn1_wickets_at)
            + innings_rows(match_id, 2, AWAY, HOME, chase_balls, chase_runs_per_ball, chase_wickets_at, target))
    df = pd.DataFrame(rows)
    df["match_won_by"], df["result_type"], df["method"] = winner, result_type, method
    df["season"] = 2015
    return df


def reasons(df):
    outline = match_outline(df)
    return outline.apply(exclusion_reason, axis=1).to_dict()


def test_compute_features_by_hand():
    # Chasing 150, 60 for 2 after 48 balls: need 90 off 72.
    f = compute_features(runs_target=150, team_runs=60, balls_bowled=48, wickets_lost=2)
    assert f["runs_left"] == 90
    assert f["balls_left"] == 72
    assert f["wickets_left"] == 8
    assert f["crr"] == pytest.approx(7.5)    # 60 runs / 8 overs
    assert f["rrr"] == pytest.approx(7.5)    # 90 runs / 12 overs


def test_compute_features_before_first_ball():
    f = compute_features(runs_target=180, team_runs=0, balls_bowled=0, wickets_lost=0)
    assert f["crr"] == 0
    assert f["rrr"] == pytest.approx(9.0)


def test_phase_buckets():
    assert [phase_of(b) for b in (0, 36, 37, 90, 91, 119)] == [
        "overs 1-6", "overs 1-6", "overs 7-15", "overs 7-15", "overs 16-20", "overs 16-20"]


def test_normal_match_is_kept():
    # Target 121; chase reaches 122 after 61 balls.
    assert reasons(make_match(1, AWAY, chase_balls=61)) == {1: None}


def test_flagged_results_are_dropped():
    df = pd.concat([
        make_match(1, "Unknown", result_type="tie"),
        make_match(2, "Unknown", result_type="no result"),
        make_match(3, AWAY, method="D/L"),
    ])
    assert reasons(df) == {1: "tie", 2: "no_result", 3: "dl_method"}


def test_reduced_first_innings_is_dropped():
    # First innings stops after 84 balls with 3 wickets down: overs were cut.
    df = make_match(1, AWAY, inn1_balls=84, inn1_wickets_at=(10, 20, 30), chase_balls=43)
    assert reasons(df) == {1: "reduced_first_innings"}


def test_all_out_first_innings_is_kept():
    df = make_match(1, AWAY, inn1_balls=84, inn1_wickets_at=range(1, 11), chase_balls=43)
    assert reasons(df) == {1: None}


def test_reduced_chase_is_dropped():
    # Chase stops at 80/2 after 40 balls, target 121 not reached, nobody all out.
    df = make_match(1, HOME, chase_balls=40, chase_wickets_at=(5, 6))
    assert reasons(df) == {1: "reduced_chase"}


def test_nine_down_ending_on_dismissal_is_kept():
    # First innings: 9th wicket falls on ball 84 (one batter couldn't bat).
    # Chase: 9th wicket falls on ball 40 with the target not reached.
    df = make_match(1, HOME, inn1_balls=84, inn1_wickets_at=range(76, 85),
                    chase_balls=40, chase_runs_per_ball=1, chase_wickets_at=range(32, 41))
    outline = match_outline(df)
    assert outline.apply(exclusion_reason, axis=1).to_dict() == {1: None}
    assert outline.loc[1, "inn1_completion"] == "nine_down_dismissal"
    assert outline.loc[1, "chase_completion"] == "nine_down_dismissal"
    assert [m["match_id"] for m in kept_by_rule(outline, "nine_down_dismissal")] == [1]


def test_nine_down_without_final_dismissal_is_dropped():
    # 9 wickets fell, but play stopped after a ball that was not a dismissal.
    df = make_match(1, AWAY, inn1_balls=84, inn1_wickets_at=range(70, 79), chase_balls=43)
    assert reasons(df) == {1: "reduced_first_innings"}


def test_final_over_glitch_is_kept_and_noted():
    # 119 legal balls but the last delivery is in the 20th over: full innings.
    df = make_match(1, AWAY, inn1_balls=119, inn1_wickets_at=(10, 20), chase_balls=61)
    outline = match_outline(df)
    assert outline.apply(exclusion_reason, axis=1).to_dict() == {1: None}
    assert [m["match_id"] for m in kept_by_rule(outline, "final_over_glitch")] == [1]


def test_nine_down_in_final_over_is_not_a_glitch():
    # Chase all out (9 down, one batter absent) on ball 117, in the 20th over.
    df = make_match(1, HOME, chase_balls=117, chase_runs_per_ball=1, chase_wickets_at=range(109, 118))
    outline = match_outline(df)
    assert outline.loc[1, "chase_completion"] == "final_over_nine_down"
    assert kept_by_rule(outline, "final_over_glitch") == []
    assert kept_by_rule(outline, "nine_down_dismissal") == []


def test_manual_exclusion_overrides_nine_down_rule():
    # Looks like a 9-down all-out (kept by the rules), but it is on the manual list.
    match_id = next(iter(MANUAL_EXCLUSIONS))
    df = make_match(match_id, AWAY, inn1_balls=72, inn1_wickets_at=range(64, 73), chase_balls=37)
    assert reasons(df) == {match_id: "manual_reduced_overs"}


def test_lost_chase_after_full_overs_is_kept():
    df = make_match(1, HOME, chase_balls=120, chase_runs_per_ball=0)
    assert reasons(df) == {1: None}


def test_states_labels_and_pre_chase_row():
    df = make_match(1, AWAY, chase_balls=61)
    outline = match_outline(df)
    states = build_states(df, outline)
    # Pre-chase state + balls 1..60; ball 61 reaches the target so it is a finished state.
    assert len(states) == 61
    assert states["balls_bowled"].tolist() == list(range(0, 61))
    assert (states["label"] == 1).all()
    assert (states["runs_left"] >= 1).all()


def test_split_keeps_every_match_in_one_split():
    df = pd.concat([make_match(i, AWAY if i % 2 else HOME, chase_balls=61) for i in range(1, 5)])
    df.loc[df["match_id"] == 1, "season"] = 2010
    df.loc[df["match_id"] == 2, "season"] = 2024
    df.loc[df["match_id"].isin([3, 4]), "season"] = 2025
    states = build_states(df, match_outline(df))
    splits = split_by_season(states)
    validate(splits)
    ids = {name: set(part["match_id"]) for name, part in splits.items()}
    assert ids == {"train": {1}, "val": {2}, "test": {3, 4}}
    assert set(SPLITS) == set(ids)
