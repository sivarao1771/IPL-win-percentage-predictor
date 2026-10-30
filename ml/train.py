"""Phase 2a: choose a model using the validation season only.

Trains every candidate on train (2008-2023) and scores it on validation (2024).
The main model is picked by SELECTION_RULE below. Never reads the test seasons.

Writes ml/build/selection.json. Run from the repo root:
  ml/.venv/Scripts/python.exe ml/train.py
"""
import json

from metrics import by_phase, paired_log_loss_diff_ci, reliability_table, scores
from models import (
    BUILD_DIR, GRID, GRID_AXES, boosted_trees, boosted_trees_with_teams, constant_baseline, load_split,
    logistic_baseline,
)

# M2 must lower validation log loss by at least 1% to justify the extra inputs.
M2_REQUIRED_IMPROVEMENT = 0.01

# A setting within this margin of the lowest validation log loss is treated as
# just as good: smaller differences are inside the noise of one 70-match season.
NEAR_BEST_MARGIN = 0.01

SELECTION_RULE = (
    f"Keep grid settings whose validation log loss is within {NEAR_BEST_MARGIN:.0%} of the lowest. "
    "Among those, prefer monotonic, then the simplest by trees x leaves, then the higher learning rate."
)


def preference_order(params):
    """Sort key for SELECTION_RULE: monotonic first, then smaller trees x leaves,
    then higher learning rate (with few trees, a low rate leaves the model under-trained)."""
    return (not params["monotonic"], params["max_iter"] * params["max_leaf_nodes"], -params["learning_rate"])


def choose_setting(grid_results):
    """Apply SELECTION_RULE to grid results that carry validation scores."""
    best = min(r["scores"]["log_loss"] for r in grid_results)
    near_best = [r for r in grid_results if r["scores"]["log_loss"] <= best * (1 + NEAR_BEST_MARGIN)]
    return min(near_best, key=lambda r: preference_order(r["params"]))


def validation_report(model, val):
    p = model.predict(val)
    return {
        "scores": scores(val["label"], p),
        "by_phase": by_phase(val["balls_bowled"], val["label"], p),
        "reliability": reliability_table(val["label"], p),
    }


def main():
    train, val = load_split("train"), load_split("val")
    print(f"Train {len(train)} rows / {train['match_id'].nunique()} matches; "
          f"validation {len(val)} rows / {val['match_id'].nunique()} matches")

    baselines = {m.name: validation_report(m.fit(train), val)
                 for m in (constant_baseline(), logistic_baseline())}

    grid_results, predictions = [], []
    for params in GRID:
        p = boosted_trees(params).fit(train).predict(val)
        grid_results.append({"params": params, "scores": scores(val["label"], p)})
        predictions.append(p)
    best_index = min(range(len(GRID)), key=lambda i: grid_results[i]["scores"]["log_loss"])
    best = grid_results[best_index]
    for result, p in zip(grid_results, predictions):
        # Information only: how much worse than the lowest, with a paired match bootstrap interval.
        result["log_loss_diff_vs_best_ci95"] = paired_log_loss_diff_ci(
            val["match_id"], val["label"], predictions[best_index], p)
        result["near_best"] = result["scores"]["log_loss"] <= best["scores"]["log_loss"] * (1 + NEAR_BEST_MARGIN)
    at_edge = {axis: best["params"][axis] for axis, values in GRID_AXES.items()
               if best["params"][axis] in (values[0], values[-1])}
    best_m1 = choose_setting(grid_results)

    m2_params = best_m1["params"]
    m2_model = boosted_trees_with_teams(m2_params, train).fit(train)
    m2 = {"params": m2_params, "scores": scores(val["label"], m2_model.predict(val))}
    m2_wins = m2["scores"]["log_loss"] <= best_m1["scores"]["log_loss"] * (1 - M2_REQUIRED_IMPROVEMENT)

    chosen_family, chosen = ("M2", m2) if m2_wins else ("M1", best_m1)
    chosen_model = (boosted_trees_with_teams(chosen["params"], train) if m2_wins
                    else boosted_trees(chosen["params"])).fit(train)

    selection = {
        "selection_metric": "validation log loss",
        "m1_rule": SELECTION_RULE,
        "m2_rule": f"M2 kept only if its validation log loss is at least "
                   f"{M2_REQUIRED_IMPROVEMENT:.0%} lower than the chosen M1",
        "baselines": baselines,
        "m1_grid": grid_results,
        "m1_lowest_log_loss": {"params": best["params"], "scores": best["scores"], "at_grid_edge": at_edge},
        "m1_near_best": sum(r["near_best"] for r in grid_results),
        "m1_not_clearly_worse": sum(r["log_loss_diff_vs_best_ci95"][0] <= 0 for r in grid_results),
        "m2": m2,
        "chosen": {"family": chosen_family, "params": chosen["params"],
                   "validation": validation_report(chosen_model, val)},
    }
    with open(BUILD_DIR / "selection.json", "w") as f:
        json.dump(selection, f, indent=2)

    print(f"Lowest log loss: {best['params']} -> {best['scores']['log_loss']:.4f} (grid edge: {at_edge or 'no'})")
    print(f"Settings within {NEAR_BEST_MARGIN:.0%} of it: {selection['m1_near_best']} of {len(GRID)} (including itself)")
    print(f"Selected setting: {best_m1['params']} -> log loss {best_m1['scores']['log_loss']:.4f}, "
          f"brier {best_m1['scores']['brier']:.4f}, acc {best_m1['scores']['accuracy']:.4f}")
    print(f"M2 (same params + team/city): log loss {m2['scores']['log_loss']:.4f} -> "
          f"{'kept' if m2_wins else 'not kept'}")
    print(f"Chosen: {chosen_family}")


if __name__ == "__main__":
    main()
