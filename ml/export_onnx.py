"""Phase 3: export the final model to ONNX and prove it matches scikit-learn.

Reads ml/build/model.joblib (saved by evaluate.py; no retraining) and writes:
  ml/artifacts/model.onnx            the model Node will run
  ml/artifacts/model_meta.json       the contract: feature order, names, input ranges, version
  ml/artifacts/parity_cases.json     raw input -> features -> probability, for the API tests

Fails if any ONNX probability differs from scikit-learn by more than 1e-5.
Run from the repo root after evaluate.py:
  ml/.venv/Scripts/python.exe ml/export_onnx.py
"""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as rt
import pandas as pd
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

from features import FEATURE_COLUMNS, TOTAL_BALLS, TOTAL_WICKETS, compute_features
from models import BUILD_DIR, SEED, load_split

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"

INPUT_NAME, OUTPUT_NAME = "features", "probabilities"
# Upper limits, fixed so the file does not change if skl2onnx's defaults do.
# skl2onnx may use lower versions; model_meta.json records the actual ones.
TARGET_OPSET = {"": 17, "ai.onnx.ml": 3}
TOLERANCE = 1e-5
MAX_TARGET = 300

# What the API accepts. Each rule has the exact error message the API returns.
INPUT_CONTRACT = {
    "target": {"min": 1, "max": MAX_TARGET, "integer": True,
               "message": f"target must be between 1 and {MAX_TARGET}"},
    "score": {"min": 0, "max": "target - 1", "integer": True,
              "message": "score must be between 0 and target - 1"},
    "ballsBowled": {"min": 0, "max": TOTAL_BALLS - 1, "integer": True,
                    "message": f"ballsBowled must be between 0 and {TOTAL_BALLS - 1}"},
    "wicketsLost": {"min": 0, "max": TOTAL_WICKETS - 1, "integer": True,
                    "message": f"wicketsLost must be between 0 and {TOTAL_WICKETS - 1}"},
}

# Hand-picked states for the parity cases: (description, target, score, balls bowled, wickets lost).
EDGE_CASES = [
    ("before the first ball", 180, 0, 0, 0),
    ("one ball left", 160, 155, 119, 4),
    ("one wicket left", 170, 120, 90, 9),
    ("one run needed", 150, 149, 100, 6),
    ("effectively impossible: 100 needed off 6 balls", 200, 100, 114, 3),
    ("very easy: 1 needed off 60 balls, no wickets lost", 120, 119, 60, 0),
    ("highest allowed target, before the first ball", 300, 0, 0, 0),
    ("lowest allowed target", 1, 0, 0, 0),
    ("last ball, last wicket, 6 needed", 175, 169, 119, 9),
    ("everything at its maximum", 300, 299, 119, 9),
]
N_SAMPLED_CASES = 190


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sort_opset_imports(onx):
    """skl2onnx lists opset imports in Python set order, which changes between
    processes (string hashing is randomised). Sorting them means the same model
    always gives the same bytes, and so the same SHA-256 and model_version."""
    imports = sorted((o.domain, o.version) for o in onx.opset_import)
    del onx.opset_import[:]
    for domain, opset_version in imports:
        onx.opset_import.add(domain=domain, version=opset_version)
    return {domain or "ai.onnx": opset_version for domain, opset_version in imports}


def feature_matrix(target, score, balls_bowled, wickets_lost):
    f = compute_features(np.asarray(target, dtype=float), np.asarray(score, dtype=float),
                         np.asarray(balls_bowled, dtype=float), np.asarray(wickets_lost, dtype=float))
    return np.column_stack([np.broadcast_to(f[c], np.shape(target)) for c in FEATURE_COLUMNS])


def edge_grid():
    """Every balls-left and wickets-left value, runs left from 1 to the target,
    at a spread of targets. Values that sit on a tree threshold are where
    float32/float64 differences would show up."""
    rows = []
    for target in (60, 100, 150, 180, 200, 220, 250, 300):
        for balls_bowled in range(TOTAL_BALLS):
            for wickets_lost in range(TOTAL_WICKETS):
                score = np.arange(0, target)
                rows.append(feature_matrix(np.full(score.shape, target), score,
                                           np.full(score.shape, balls_bowled), np.full(score.shape, wickets_lost)))
    return np.vstack(rows)


def max_parity_gap(model, session, X):
    X32 = X.astype(np.float32)
    p_sklearn = model.predict_proba(X32)[:, 1]
    p_onnx = session.run([OUTPUT_NAME], {INPUT_NAME: X32})[0][:, 1]
    return float(np.abs(p_sklearn - p_onnx).max())


def parity_cases(session, test):
    picks = test.sample(n=N_SAMPLED_CASES, random_state=SEED)
    raw = [(f"test row: match {int(r.match_id)}", int(r.runs_target), int(r.runs_target - r.runs_left),
            int(TOTAL_BALLS - r.balls_left), int(TOTAL_WICKETS - r.wickets_left)) for r in picks.itertuples()]
    cases = []
    for description, target, score, balls_bowled, wickets_lost in EDGE_CASES + raw:
        features = compute_features(target, score, balls_bowled, wickets_lost)
        x = np.array([[float(features[c]) for c in FEATURE_COLUMNS]], dtype=np.float32)
        p = session.run([OUTPUT_NAME], {INPUT_NAME: x})[0][0, 1]
        cases.append({
            "description": description,
            "input": {"target": target, "score": score, "ballsBowled": balls_bowled, "wicketsLost": wickets_lost},
            "features": {c: float(features[c]) for c in FEATURE_COLUMNS},
            "winProbability": float(p),
        })
    return cases


def main():
    model = joblib.load(BUILD_DIR / "model.joblib")
    onx = to_onnx(model, initial_types=[(INPUT_NAME, FloatTensorType([None, len(FEATURE_COLUMNS)]))],
                  target_opset=TARGET_OPSET, options={id(model): {"zipmap": False}},
                  final_types=[("label", None), (OUTPUT_NAME, None)])
    opsets_in_file = sort_opset_imports(onx)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "model.onnx"
    model_path.write_bytes(onx.SerializeToString())
    session = rt.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    test, val = load_split("test"), load_split("val")
    gaps = {
        "test rows": max_parity_gap(model, session, test[FEATURE_COLUMNS].to_numpy()),
        "validation rows": max_parity_gap(model, session, val[FEATURE_COLUMNS].to_numpy()),
        "edge grid": max_parity_gap(model, session, edge_grid()),
    }
    for name, gap in gaps.items():
        print(f"Max |sklearn - onnx| on {name}: {gap:.3e}")
    worst = max(gaps.values())
    if worst > TOLERANCE:
        raise SystemExit(f"Parity check FAILED: {worst:.3e} > {TOLERANCE:.0e}")

    cases = parity_cases(session, test)
    (ARTIFACTS_DIR / "parity_cases.json").write_text(json.dumps({"tolerance": TOLERANCE, "cases": cases}, indent=2))

    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text())
    digest = sha256_of(model_path)
    meta = {
        "model_version": f"ipl-chase-hgb-{digest[:12]}",
        "model_sha256": digest,
        "description": "P(chasing team wins) in an IPL second innings, from the current match state.",
        "input": {"name": INPUT_NAME, "dtype": "float32", "shape": [None, len(FEATURE_COLUMNS)],
                  "feature_order": FEATURE_COLUMNS},
        "output": {"name": OUTPUT_NAME, "dtype": "float32", "shape": [None, 2],
                   "win_probability_column": 1},
        "features": {
            "runs_left": "target - score",
            "balls_left": f"{TOTAL_BALLS} - ballsBowled",
            "wickets_left": f"{TOTAL_WICKETS} - wicketsLost",
            "runs_target": "target",
            "crr": "score * 6 / ballsBowled, or 0 when ballsBowled is 0",
            "rrr": "runsLeft * 6 / ballsLeft",
            "note": "Compute in float64 in exactly this order, then convert to float32.",
        },
        "input_contract": INPUT_CONTRACT,
        "training_range_note": "Training targets ranged from 68 to 288. Outside that range the trees "
                               "repeat their nearest learned values; they do not extrapolate.",
        # skl2onnx lowers each opset to the minimum the model needs, so record what
        # is actually in the file rather than the requested maximum.
        "opset": opsets_in_file,
        "parity": {"tolerance": TOLERANCE, "max_abs_diff": gaps},
        "provenance": {
            "trained_on": metrics["final_model"]["trained_on"],
            "params": metrics["final_model"]["params"],
            "training_code_commit": metrics["git"]["commit"],
            "dataset_sha256": metrics["dataset"]["sha256"],
            "packages": {p: version(p) for p in ["scikit-learn", "skl2onnx", "onnx", "onnxruntime"]},
        },
    }
    (ARTIFACTS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {model_path.name} ({model_path.stat().st_size:,} bytes), model_meta.json, "
          f"parity_cases.json ({len(cases)} cases). Version {meta['model_version']}")


if __name__ == "__main__":
    main()
