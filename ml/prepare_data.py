"""Phase 1: turn the raw ball-by-ball CSV into clean, season-split chase states.

Reads data/IPL.csv (never modifies it) and writes:
  ml/build/{train,val,test}.csv          one row per chase state, with label
  ml/reports/data_summary.json           dataset version, drop counts, split sizes
  ml/reports/excluded_matches.csv        every dropped match and why, for spot checks

Run from the repo root:  ml/.venv/Scripts/python.exe ml/prepare_data.py
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from features import TOTAL_BALLS, TOTAL_WICKETS, compute_features

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "IPL.csv"
BUILD_DIR = ROOT / "ml" / "build"
REPORTS_DIR = ROOT / "ml" / "reports"

RAW_COLUMNS = [
    "match_id", "date", "innings", "batting_team", "bowling_team", "city", "over",
    "team_runs", "team_balls", "team_wicket", "player_out", "runs_target",
    "match_won_by", "result_type", "method",
]
LAST_OVER = TOTAL_BALLS // 6 - 1  # overs are numbered from 0 in the data

# Same franchise, renamed. Exact-name mapping (not substring replace), so
# "Rising Pune Supergiants" is not turned into "Rising Pune Supergiantss".
TEAM_RENAMES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}
CITY_RENAMES = {"Bangalore": "Bengaluru"}

# Shortened matches the automatic rules can't detect: the first innings ended on
# a 9th-wicket dismissal exactly at the reduced overs limit, so it looks like an
# all-out. Each entry was checked by hand.
MANUAL_EXCLUSIONS = {
    336012: "Reduced to 18 overs a side before play (scorecard: "
            "https://espncricinfo.com/ci/engine/match/336012.html). "
            "RCB 122/9 ended at exactly 18.0 overs.",
    548307: "12-over match: KKR 97/9 ended on a run-out at exactly 12.0 overs and the "
            "chase finished at 11.1 overs, both inside a 12-over limit.",
}

# Inclusive season ranges. The test seasons are only looked at once, at the end.
SPLITS = {"train": (2008, 2023), "val": (2024, 2024), "test": (2025, 2026)}

OUTPUT_COLUMNS = [
    "match_id", "season", "batting_team", "bowling_team", "city", "balls_bowled",
    "runs_left", "balls_left", "wickets_left", "runs_target", "crr", "rrr", "label",
]


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path):
    df = pd.read_csv(path, usecols=RAW_COLUMNS, low_memory=False)
    for col in ["batting_team", "bowling_team", "match_won_by"]:
        df[col] = df[col].replace(TEAM_RENAMES)
    df["city"] = df["city"].replace(CITY_RENAMES)
    df["season"] = pd.to_datetime(df["date"]).dt.year
    return df


def innings_completion(balls, wickets, last_over, last_ball_dismissal):
    """Why an innings counts as played in full, or None if it looks shortened.

    The order matters: a match is only "kept by the 9-wicket rule" when none of
    the stronger reasons apply.
    """
    if balls >= TOTAL_BALLS:
        return "full_overs"
    if wickets >= TOTAL_WICKETS:
        return "all_out"
    # All out with one batter unable to bat: the 9th wicket ends the innings.
    nine_down = wickets >= TOTAL_WICKETS - 1 and last_ball_dismissal
    if last_over == LAST_OVER:
        # Both rules agree for a 9-down finish in the 20th over. Otherwise all 20
        # overs were bowled but the data counts fewer legal balls (data glitch).
        return "final_over_nine_down" if nine_down else "final_over_glitch"
    if nine_down:
        return "nine_down_dismissal"
    return None


def innings_summary(df, innings):
    inn = df[df["innings"] == innings]
    grouped = inn.groupby("match_id")
    summary = grouped.agg(runs=("team_runs", "max"), balls=("team_balls", "max"),
                          wickets=("team_wicket", "max"))
    last = grouped.tail(1).set_index("match_id")
    summary["last_over"] = last["over"]
    summary["last_ball_dismissal"] = last["player_out"].notna()
    summary["completion"] = [
        innings_completion(*row) for row in
        summary[["balls", "wickets", "last_over", "last_ball_dismissal"]].itertuples(index=False)]
    return summary


def match_outline(df):
    """One row per match: how each innings ended, plus result metadata."""
    first = df.groupby("match_id").first()
    inn1 = innings_summary(df, 1)
    chase = innings_summary(df, 2)
    chase_info = df[df["innings"] == 2].groupby("match_id").agg(
        chase_team=("batting_team", "first"), target=("runs_target", "first"))
    chase = chase.join(chase_info)
    chase.loc[chase["runs"] >= chase["target"], "completion"] = "target_reached"

    outline = first[["date", "season", "batting_team", "bowling_team", "match_won_by", "result_type", "method"]]
    outline = outline.rename(columns={"batting_team": "inn1_team", "bowling_team": "inn1_opponent"})
    outline = outline.join(inn1.add_prefix("inn1_"))
    outline = outline.join(chase.drop(columns=["chase_team", "target"]).add_prefix("chase_"))
    return outline.join(chase[["chase_team", "target"]])


def exclusion_reason(m):
    """Why a match can't be used, or None. The first matching reason wins, so each
    match is counted once."""
    if m["result_type"] == "no result":
        return "no_result"
    if m["result_type"] == "tie":
        return "tie"
    if m["method"] == "D/L":
        return "dl_method"
    if m.name in MANUAL_EXCLUSIONS:
        return "manual_reduced_overs"
    if pd.isna(m["inn1_balls"]):
        return "no_first_innings"
    if pd.isna(m["chase_balls"]):
        return "no_chase"
    if pd.isna(m["inn1_completion"]):
        return "reduced_first_innings"
    if pd.isna(m["chase_completion"]):
        return "reduced_chase"
    if m["match_won_by"] not in (m["inn1_team"], m["inn1_opponent"]):
        return "unknown_winner"
    chase_won = m["match_won_by"] == m["chase_team"]
    if chase_won != (m["chase_completion"] == "target_reached"):
        return "result_inconsistent_with_score"
    return None


def build_states(df, outline):
    """One row per chase state: the state after every delivery, plus the state
    before the first ball. Finished states (result already known) are removed."""
    chase = df[df["innings"] == 2]
    first_rows = chase.groupby("match_id", sort=False).head(1).copy()
    first_rows[["team_runs", "team_balls", "team_wicket"]] = 0
    states = pd.concat([first_rows, chase], ignore_index=True)

    states = states.rename(columns={"team_balls": "balls_bowled"})
    states = states.assign(**compute_features(
        states["runs_target"], states["team_runs"], states["balls_bowled"], states["team_wicket"]))
    states = states[(states["runs_left"] > 0) & (states["balls_left"] > 0) & (states["wickets_left"] > 0)]

    states["label"] = (outline.loc[states["match_id"], "match_won_by"].to_numpy()
                       == states["batting_team"].to_numpy()).astype(int)
    return states[OUTPUT_COLUMNS].reset_index(drop=True)


def split_by_season(states):
    return {name: states[states["season"].between(lo, hi)].reset_index(drop=True)
            for name, (lo, hi) in SPLITS.items()}


def kept_by_rule(kept, completion):
    """Kept matches where at least one innings only counted as complete because of
    `completion`. These are listed in the summary so they can be checked by hand."""
    hit = kept[(kept["inn1_completion"] == completion) | (kept["chase_completion"] == completion)]
    return [
        {
            "match_id": int(match_id),
            "date": str(m["date"]),
            "teams": f"{m['inn1_team']} v {m['inn1_opponent']}",
            "innings": [name for name in ("inn1", "chase") if m[f"{name}_completion"] == completion],
            "inn1": f"{m['inn1_runs']}/{m['inn1_wickets']} in {m['inn1_balls']} balls",
            "chase": f"{m['chase_runs']}/{m['chase_wickets']} in {m['chase_balls']} balls (target {m['target']:.0f})",
        }
        for match_id, m in hit.iterrows()
    ]


def validate(splits):
    """Stop with an error if any rule of the split is broken."""
    seen = {}
    for name, part in splits.items():
        lo, hi = SPLITS[name]
        assert part["season"].between(lo, hi).all(), f"{name}: season outside {lo}-{hi}"
        assert not part.isna().any().any(), f"{name}: missing values"
        assert part["balls_left"].between(1, TOTAL_BALLS).all(), f"{name}: balls_left out of range"
        assert part["wickets_left"].between(1, TOTAL_WICKETS).all(), f"{name}: wickets_left out of range"
        assert (part["runs_left"] >= 1).all(), f"{name}: runs_left < 1"
        assert (part.groupby("match_id")["label"].nunique() == 1).all(), f"{name}: label varies within a match"
        for match_id in part["match_id"].unique():
            assert match_id not in seen, f"match {match_id} in both {seen[match_id]} and {name}"
            seen[match_id] = name


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    df = load_raw(args.data)
    outline = match_outline(df)
    outline["reason"] = outline.apply(exclusion_reason, axis=1)
    excluded = outline[outline["reason"].notna()]
    kept = outline[outline["reason"].isna()]
    kept_ids = kept.index

    states = build_states(df[df["match_id"].isin(kept_ids)], outline)
    splits = split_by_season(states)
    validate(splits)

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(BUILD_DIR / f"{name}.csv", index=False)
    excluded.drop(columns=["result_type", "method"]).to_csv(REPORTS_DIR / "excluded_matches.csv")

    summary = {
        "dataset": {
            "file": args.data.name,
            "sha256": sha256_of(args.data),
            "rows": int(len(df)),
            "matches": int(df["match_id"].nunique()),
            "first_date": str(df["date"].min()),
            "last_date": str(df["date"].max()),
        },
        "excluded_matches": {
            reason: {"count": int(len(group)), "match_ids": [int(i) for i in group.index]}
            for reason, group in excluded.groupby("reason")
        },
        "manual_exclusions": {str(k): v for k, v in MANUAL_EXCLUSIONS.items()},
        "matches_kept": int(len(kept_ids)),
        "kept_with_notes": {
            "final_over_glitch": {
                "note": "All 20 overs were bowled (last delivery in the 20th over), but the data "
                        "counts fewer than 120 legal balls and fewer than 10 wickets. Kept.",
                "matches": kept_by_rule(kept, "final_over_glitch"),
            },
            "nine_down_dismissal": {
                "note": "Innings ended on a dismissal with 9 wickets down before the 20th over: "
                        "all out with one batter unable to bat. Kept only because of this rule; "
                        "listed for manual checking.",
                "matches": kept_by_rule(kept, "nine_down_dismissal"),
            },
        },
        "splits": {
            name: {
                "seasons": f"{SPLITS[name][0]}-{SPLITS[name][1]}",
                "matches": int(part["match_id"].nunique()),
                "rows": int(len(part)),
                "chase_win_rate_by_match": round(float(part.groupby("match_id")["label"].first().mean()), 4),
            }
            for name, part in splits.items()
        },
    }
    with open(REPORTS_DIR / "data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    d = summary["dataset"]
    print(f"Dataset {d['file']}: {d['rows']} rows, {d['matches']} matches, sha256 {d['sha256'][:12]}...")
    print("Excluded matches:")
    for reason, info in summary["excluded_matches"].items():
        print(f"  {reason:32s} {info['count']:4d}  {info['match_ids']}")
    print(f"Matches kept: {summary['matches_kept']}")
    for rule, info in summary["kept_with_notes"].items():
        print(f"Kept with note '{rule}': {len(info['matches'])}")
        for m in info["matches"]:
            print(f"  {m['match_id']} {m['date']} {m['teams']} [{', '.join(m['innings'])}] "
                  f"inn1 {m['inn1']}; chase {m['chase']}")
    for name, info in summary["splits"].items():
        print(f"  {name:5s} {info['seasons']}: {info['matches']:4d} matches, {info['rows']:6d} rows, "
              f"chase win rate {info['chase_win_rate_by_match']}")


if __name__ == "__main__":
    main()
