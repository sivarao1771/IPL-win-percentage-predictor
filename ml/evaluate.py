"""Phase 2b: the one-time final evaluation on the test seasons (2025-2026).

Retrains the baselines and the model chosen by train.py on train + validation
(2008-2024), scores them once on test, and writes the report. This is the only
script that reads test.csv.

Writes ml/build/model.joblib (for the Phase 3 ONNX export), ml/reports/metrics.json
and ml/reports/results.md. Run from the repo root after train.py:
  ml/.venv/Scripts/python.exe ml/evaluate.py
"""
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from features import CATEGORY_COLUMNS, FEATURE_COLUMNS
from metrics import bootstrap_ci, by_phase, paired_log_loss_diff_ci, reliability_table, scores
from models import (
    BUILD_DIR, SEED, Model, boosted_trees_with_teams, build_final, constant_baseline, load_split,
    logistic_baseline,
)

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
PACKAGES = ["numpy", "pandas", "scikit-learn"]


def git_commit():
    root = Path(__file__).resolve().parents[1]
    run = lambda *args: subprocess.run(["git", *args], cwd=root, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "uncommitted_changes": bool(run("status", "--porcelain"))}


def old_notebook_forest():
    """The Random Forest settings from the old notebook (cell 36), with one-hot
    team/city like the old preprocessor. Deep trees with small leaves can
    memorise individual matches, which is what a random row split rewards."""
    encoder = ColumnTransformer([
        ("state", "passthrough", FEATURE_COLUMNS),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORY_COLUMNS),
    ])
    forest = RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_split=10, min_samples_leaf=5,
                                    class_weight="balanced", random_state=SEED, n_jobs=-1)
    return Model("Old notebook Random Forest", make_pipeline(encoder, forest),
                 FEATURE_COLUMNS + CATEGORY_COLUMNS, uses_categories=True)


def leakage_demo(all_states, selection):
    """Same model and data, only the way test rows are chosen changes.
    Shows how much a random row split flatters the score."""
    final_params = selection["chosen"]["params"]
    results = []
    row_train, row_test = train_test_split(all_states, test_size=0.2, random_state=SEED)
    groups = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(groups.split(all_states, groups=all_states["match_id"]))
    splits = [
        ("Random rows (old notebook method)", row_train, row_test),
        ("Random matches", all_states.iloc[train_idx], all_states.iloc[test_idx]),
    ]
    for split_name, train, test in splits:
        shared = len(set(train["match_id"]) & set(test["match_id"]))
        for feature_name, make in [
            ("match state", lambda tr: build_final({"chosen": {"family": "M1", "params": final_params}}, tr)),
            ("match state + team/city", lambda tr: boosted_trees_with_teams(final_params, tr)),
        ]:
            p = make(train).fit(train).predict(test)
            results.append({"split": split_name, "model": "Final model settings", "features": feature_name,
                            "test_matches_also_in_train": shared,
                            "test_matches": int(test["match_id"].nunique()),
                            **scores(test["label"], p)})
    p = old_notebook_forest().fit(row_train).predict(row_test)
    results.append({"split": "Random rows (old notebook method)", "model": "Old notebook Random Forest",
                    "features": "match state + team/city",
                    "test_matches_also_in_train": len(set(row_train["match_id"]) & set(row_test["match_id"])),
                    "test_matches": int(row_test["match_id"].nunique()),
                    **scores(row_test["label"], p)})
    return results


def main():
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    selection = json.loads((BUILD_DIR / "selection.json").read_text())
    data_summary = json.loads((REPORTS_DIR / "data_summary.json").read_text())
    train_val = pd.concat([train, val], ignore_index=True)

    models = [constant_baseline(), logistic_baseline(), build_final(selection, train_val)]
    test_results, test_predictions = {}, {}
    for model in models:
        p = model.fit(train_val).predict(test)
        test_predictions[model.name] = p
        test_results[model.name] = {
            "scores": scores(test["label"], p),
            "ci95": bootstrap_ci(test["match_id"], test["label"], p),
            "by_phase": by_phase(test["balls_bowled"], test["label"], p),
            "reliability": reliability_table(test["label"], p),
        }
    final, logistic = models[-1], models[1]
    joblib.dump(final.estimator, BUILD_DIR / "model.joblib")
    final_vs_logistic = {
        "reference": logistic.name,
        "log_loss_diff_ci95": paired_log_loss_diff_ci(
            test["match_id"], test["label"], test_predictions[logistic.name], test_predictions[final.name]),
    }

    print("Leakage demonstration (this takes a minute)...")
    demo = leakage_demo(pd.concat([train, val, test], ignore_index=True), selection)

    metrics = {
        "dataset": data_summary["dataset"],
        "seed": SEED,
        "packages": {"python": platform.python_version(), **{p: version(p) for p in PACKAGES}},
        "git": git_commit(),
        "splits": data_summary["splits"],
        "selection": selection,
        "test": test_results,
        "test_final_vs_logistic": final_vs_logistic,
        "final_model": {"name": final.name, "columns": final.columns, "params": selection["chosen"]["params"],
                        "trained_on": "train + validation (2008-2024)"},
        "leakage_demo": demo,
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (REPORTS_DIR / "results.md").write_text(render_report(metrics), encoding="utf-8")
    print("Wrote ml/reports/metrics.json and ml/reports/results.md")


# ---------- Markdown report ----------

def pct(x):
    return f"{x * 100:.2f}%"


def num(x):
    return f"{x:.4f}"


def score_row(name, s, ci=None):
    cells = [pct(s["accuracy"]), num(s["log_loss"]), num(s["brier"])]
    if ci:
        cells = [f"{pct(s['accuracy'])} ({pct(ci['accuracy'][0])}-{pct(ci['accuracy'][1])})",
                 f"{num(s['log_loss'])} ({num(ci['log_loss'][0])}-{num(ci['log_loss'][1])})",
                 f"{num(s['brier'])} ({num(ci['brier'][0])}-{num(ci['brier'][1])})"]
    return f"| {name} | {s['rows']:,} | " + " | ".join(cells) + " |"


def reliability_rows(table):
    lines = ["| Predicted | Rows | Mean predicted | Actual win rate |", "|---|---|---|---|"]
    for b in table:
        if b["rows"]:
            lines.append(f"| {b['bucket']} | {b['rows']:,} | {pct(b['mean_predicted'])} | {pct(b['actual_win_rate'])} |")
        else:
            lines.append(f"| {b['bucket']} | 0 | - | - |")
    return lines


def calibration_observation(table):
    filled = [b for b in table if b["rows"]]
    above = sum(b["actual_win_rate"] > b["mean_predicted"] for b in filled)
    return (f"In {above} of {len(filled)} non-empty buckets, the actual chase win rate was higher "
            f"than the mean prediction.")


def summary(m):
    test, final_name = m["test"], m["final_model"]["name"]
    ref_name = m["test_final_vs_logistic"]["reference"]
    f, b = test[final_name]["scores"], test[ref_name]["scores"]
    low, high = m["test_final_vs_logistic"]["log_loss_diff_ci95"]
    verdict = "beat" if f["log_loss"] < b["log_loss"] else "did not beat"
    noise = ("includes 0, so the difference is within the noise" if low <= 0 <= high
             else "excludes 0, so the difference is larger than the noise")
    phases = "; ".join(
        f"{phase}: {num(test[final_name]['by_phase'][phase]['log_loss'])} vs "
        f"{num(test[ref_name]['by_phase'][phase]['log_loss'])}"
        for phase in test[final_name]["by_phase"])
    base = next(r for name, r in test.items() if name.startswith("B0"))["scores"]
    return [
        f"**Summary.** On the test seasons ({m['splits']['test']['seasons']}, "
        f"{m['splits']['test']['matches']} matches), the final model (gradient-boosted trees) {verdict} "
        f"the logistic-regression baseline B1 on log loss, the selection metric: final vs B1 accuracy "
        f"{pct(f['accuracy'])} vs {pct(b['accuracy'])}, log loss {num(f['log_loss'])} vs {num(b['log_loss'])}, "
        f"Brier {num(f['brier'])} vs {num(b['brier'])}. The paired 95% interval for the log-loss difference "
        f"(final - B1, resampling whole test matches) is {low:+.4f} to {high:+.4f}, which {noise}.",
        "",
        f"By phase, log loss final vs B1: {phases}. Both models are far better than always predicting "
        f"the base rate (log loss {num(base['log_loss'])}).",
        "",
    ]


def render_report(m):
    d, sel, test = m["dataset"], m["selection"], m["test"]
    final_name = m["final_model"]["name"]
    header = "| Model | Rows | Accuracy | Log loss | Brier |\n|---|---|---|---|---|"
    out = [
        "# IPL chase win probability: results",
        "",
        *summary(m),
        "Generated by `ml/evaluate.py`. Every number below is copied from `metrics.json`, which keeps "
        "full precision; tables show 4 decimal places (accuracy as a percentage with 2).",
        "",
        "**Metrics.** Accuracy: share of rows where the side of 50% matched the result. "
        "Log loss: penalty for the probability given to what actually happened; lower is better, "
        "a constant 50% guess scores 0.6931. Brier score: mean squared gap between probability "
        "and result; lower is better, a constant 50% guess scores 0.25.",
        "",
        "## 1. Data and reproducibility",
        "",
        f"- Dataset: `{d['file']}`, {d['rows']:,} rows, {d['matches']:,} matches, "
        f"{d['first_date']} to {d['last_date']}",
        f"- SHA-256: `{d['sha256']}`",
        f"- Seed: {m['seed']}; Python {m['packages']['python']}, "
        + ", ".join(f"{k} {v}" for k, v in m["packages"].items() if k != "python"),
        f"- Code: commit `{m['git']['commit'][:10]}`"
        + (" (with uncommitted changes)" if m["git"]["uncommitted_changes"] else ""),
        "- One row per chase state (after each ball, plus before the first ball). Matches with no "
        "result, ties, D/L and shortened matches are excluded; see `data_summary.json`.",
        "",
        "| Split | Seasons | Matches | Rows | Chases won (by match) |",
        "|---|---|---|---|---|",
        *[f"| {name} | {s['seasons']} | {s['matches']} | {s['rows']:,} | {pct(s['chase_win_rate_by_match'])} |"
          for name, s in m["splits"].items()],
        "",
        "## 2. Final results on the test seasons (2025-2026)",
        "",
        "All models trained on 2008-2024 and scored once on 2025-2026. Brackets: 95% confidence "
        "interval from 1,000 resamples of whole test matches.",
        "",
        header,
        *[score_row(name, r["scores"], r["ci95"]) for name, r in test.items()],
        "",
        f"Final model: gradient-boosted trees on {', '.join(f'`{c}`' for c in m['final_model']['columns'])}. "
        f"Settings: `{m['final_model']['params']}`.",
        "",
        "### By phase of the chase (test)",
        "",
    ]
    for phase in next(iter(test.values()))["by_phase"]:
        out += [f"**{phase.capitalize()}**", "", header,
                *[score_row(name, r["by_phase"][phase]) for name, r in test.items()], ""]

    out += [
        "### Calibration (test, final model)",
        "",
        "If the model is well calibrated, the actual win rate in each row is close to the mean prediction.",
        "",
        *reliability_rows(test[final_name]["reliability"]),
        "",
        calibration_observation(test[final_name]["reliability"]),
        "",
        "## 3. Model selection (validation season 2024 only)",
        "",
        f"Models trained on 2008-2023 and compared on 2024. Selection metric: {sel['selection_metric']}. "
        f"{sel['m2_rule']}.",
        "",
        header,
        *[score_row(name, r["scores"]) for name, r in sel["baselines"].items()],
        score_row(f"Chosen ({sel['chosen']['family']})", sel["chosen"]["validation"]["scores"]),
        "",
        f"**How the main model was picked.** {sel['m1_rule']}",
        "",
        f"The lowest validation log loss was {num(sel['m1_lowest_log_loss']['scores']['log_loss'])} "
        f"(`{sel['m1_lowest_log_loss']['params']}`), at the edge of the grid for "
        f"{', '.join(sel['m1_lowest_log_loss']['at_grid_edge']) or 'no setting'}. One season of "
        f"{m['splits']['val']['matches']} matches cannot separate most settings: for "
        f"{sel['m1_not_clearly_worse']} of {len(sel['m1_grid'])} the paired 95% interval of the difference "
        "from the lowest includes 0. So the edge carries little information, and the rule picks among "
        f"the {sel['m1_near_best']} settings within the margin instead of trusting the single lowest. "
        "The paired intervals are shown for information; they are not used to select, because with this "
        "little data they also fail to rule out clearly weaker models.",
        "",
        "**Main-model grid** (gradient-boosted trees on match state; min_samples_leaf = 100 for all; "
        "sorted by validation log loss; **bold** = chosen). Last column: 95% interval for "
        "(log loss of this setting - lowest log loss), paired bootstrap over 2024 matches.",
        "",
        "| Learning rate | Trees | Max leaves | Monotonic | Accuracy | Log loss | Brier | Within 1% | "
        "Difference from lowest (95% CI) |",
        "|---|---|---|---|---|---|---|---|---|",
        *[("| **{}** | **{}** | **{}** | **{}** | **{}** | **{}** | **{}** | **{}** | **{}** |"
           if g["params"] == sel["chosen"]["params"] else "| {} | {} | {} | {} | {} | {} | {} | {} | {} |").format(
              g["params"]["learning_rate"], g["params"]["max_iter"], g["params"]["max_leaf_nodes"],
              "yes" if g["params"]["monotonic"] else "no", pct(g["scores"]["accuracy"]),
              num(g["scores"]["log_loss"]), num(g["scores"]["brier"]), "yes" if g["near_best"] else "no",
              f"{g['log_loss_diff_vs_best_ci95'][0]:+.4f} to {g['log_loss_diff_vs_best_ci95'][1]:+.4f}")
          for g in sorted(sel["m1_grid"], key=lambda g: g["scores"]["log_loss"])],
        "",
        f"Team/city experiment (M2, best grid settings + one-hot batting team, bowling team, city): "
        f"log loss {num(sel['m2']['scores']['log_loss'])}, Brier {num(sel['m2']['scores']['brier'])}, "
        f"accuracy {pct(sel['m2']['scores']['accuracy'])}. "
        f"{'Kept' if sel['chosen']['family'] == 'M2' else 'Not kept'}.",
        "",
        "### Calibration (validation, chosen model trained on 2008-2023)",
        "",
        *reliability_rows(sel["chosen"]["validation"]["reliability"]),
        "",
        calibration_observation(sel["chosen"]["validation"]["reliability"]),
        "",
        "**Possible cause (untested hypothesis).** The Impact Player rule, introduced in 2023, lets "
        "each team add a substitute, which in effect lengthens the batting order and could make "
        "chases easier than in the 2008-2022 seasons that dominate training. This has not been "
        "tested, and with 70 validation matches part of the gap may be noise. No correction was "
        "applied; the model is reported as is.",
        "",
        "## 4. Leakage demonstration",
        "",
        "**Not a result of this project.** It shows how the old notebook's evaluation inflated scores. "
        "All rows except the last use all seasons' data (2008-2026) with an 80/20 split. With a random "
        "row split, every test match also has rows in training, so a model can recognise the match "
        "instead of predicting it. How much it gains depends on whether the model can memorise: the "
        "final model's settings (small trees, at least 100 rows per leaf) can do so far less than the "
        "old notebook's Random Forest (150 trees, depth 20, at least 5 rows per leaf).",
        "",
        "| Test rows picked by | Model | Features | Test matches also in training | Accuracy | Log loss | Brier |",
        "|---|---|---|---|---|---|---|",
        *[f"| {r['split']} | {r['model']} | {r['features']} | {r['test_matches_also_in_train']} of "
          f"{r['test_matches']} | {pct(r['accuracy'])} | {num(r['log_loss'])} | {num(r['brier'])} |"
          for r in m["leakage_demo"]],
        f"| Seasons (honest, section 2) | Final model | match state | 0 of {m['splits']['test']['matches']} | "
        f"{pct(test[final_name]['scores']['accuracy'])} | {num(test[final_name]['scores']['log_loss'])} | "
        f"{num(test[final_name]['scores']['brier'])} |",
        "",
        "## 5. Future work",
        "",
        "The test seasons 2025-2026 have now been used. Any change below must be judged on a new, "
        "untouched test season (2027 or later), not on 2025-2026 again.",
        "",
        "- **Recency weighting:** give recent seasons more weight in training, to test whether the "
        "calibration gap in sections 2-3 comes from the game changing over time (for example the "
        "Impact Player rule).",
        "- **Multi-season (rolling) validation:** choose settings by training on all seasons before "
        "each of several recent seasons and scoring on that season, instead of on 2024 alone. "
        "About 400 validation matches instead of 70 would make differences between settings "
        "measurable.",
        "",
        "## 6. Reproduce",
        "",
        "```",
        "python -m venv ml/.venv",
        "ml/.venv/Scripts/python.exe -m pip install -r ml/requirements.txt",
        "ml/.venv/Scripts/python.exe ml/prepare_data.py",
        "ml/.venv/Scripts/python.exe ml/train.py",
        "ml/.venv/Scripts/python.exe ml/evaluate.py",
        "```",
        "",
    ]
    return "\n".join(out)


if __name__ == "__main__":
    main()
