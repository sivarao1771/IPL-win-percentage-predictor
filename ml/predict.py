"""Win probability for the chasing team, from the exported model.

As a function:
    from predict import predict_win_probability
    predict_win_probability(target=180, score=95, balls_bowled=66, wickets_lost=3)

From the command line (repo root):
    ml/.venv/Scripts/python.exe ml/predict.py --target 180 --score 95 --balls 66 --wickets 3

Uses the committed ml/artifacts/model.onnx, so it works from a fresh clone without
the dataset. Input rules and messages come from ml/artifacts/model_meta.json, the
same contract the API enforces.
"""
import argparse
import hashlib
import json
import sys
from functools import lru_cache
from numbers import Integral
from pathlib import Path
from typing import NamedTuple

import numpy as np
import onnxruntime as rt

from features import FEATURE_COLUMNS, compute_features

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
INVALID_INPUT_EXIT_CODE = 2


class InvalidInputError(ValueError):
    """Raised with every problem found in the input, not just the first."""

    def __init__(self, errors):
        super().__init__("; ".join(errors))
        self.errors = list(errors)


class LoadedModel(NamedTuple):
    session: rt.InferenceSession
    meta: dict


def _read_required(path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return path.read_bytes()


def load_model(model_dir=ARTIFACTS_DIR):
    """Load and check the model. Raises if a file is missing, the model does not
    match the SHA-256 in model_meta.json, or its inputs, outputs or feature order
    differ from what this code expects."""
    model_dir = Path(model_dir)
    meta = json.loads(_read_required(model_dir / "model_meta.json"))
    # Hash and load the same bytes, so the checked file is the one that runs.
    model_bytes = _read_required(model_dir / "model.onnx")

    digest = hashlib.sha256(model_bytes).hexdigest()
    if digest != meta["model_sha256"]:
        raise ValueError(f"model.onnx SHA-256 {digest} does not match model_meta.json ({meta['model_sha256']})")
    if meta["input"]["feature_order"] != FEATURE_COLUMNS:
        raise ValueError(f"Feature order in model_meta.json {meta['input']['feature_order']} "
                         f"does not match features.py {FEATURE_COLUMNS}")

    session = rt.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    if meta["input"]["name"] not in [i.name for i in session.get_inputs()]:
        raise ValueError(f"Model input {meta['input']['name']!r} not found in model.onnx")
    if meta["output"]["name"] not in [o.name for o in session.get_outputs()]:
        raise ValueError(f"Model output {meta['output']['name']!r} not found in model.onnx")
    return LoadedModel(session, meta)


@lru_cache(maxsize=1)
def default_model():
    """The committed model, loaded on first use and then reused."""
    return load_model(ARTIFACTS_DIR)


def _is_integer(value):
    # bool is a subclass of int in Python, but True is not a valid score.
    return isinstance(value, Integral) and not isinstance(value, bool)


def validate(inputs, contract):
    """Check inputs (keyed by contract name) and return a list of error messages.
    A wrong type gets a type message; only integers are range-checked. The
    score-vs-target rule is skipped when target itself is invalid."""
    errors = []
    target_rule = contract["target"]
    target = inputs["target"]
    target_is_valid = _is_integer(target) and target_rule["min"] <= target <= target_rule["max"]
    for name, rule in contract.items():
        value = inputs[name]
        if not _is_integer(value):
            errors.append(f"{name} must be an integer")
            continue
        if rule["max"] == "target - 1":
            if value < rule["min"] or (target_is_valid and value > target - 1):
                errors.append(rule["message"])
        elif not rule["min"] <= value <= rule["max"]:
            errors.append(rule["message"])
    return errors


def predict_win_probability(target, score, balls_bowled, wickets_lost, model=None):
    """Probability (0 to 1, not rounded) that the chasing team wins from this state.

    target: runs needed to win (first-innings score + 1), 1 to 300
    score: runs scored so far, 0 to target - 1
    balls_bowled: legal balls bowled so far (wides and no-balls don't count), 0 to 119
    wickets_lost: 0 to 9

    Raises InvalidInputError listing every invalid input.
    """
    session, meta = model or default_model()
    inputs = {"target": target, "score": score, "ballsBowled": balls_bowled, "wicketsLost": wickets_lost}
    errors = validate(inputs, meta["input_contract"])
    if errors:
        raise InvalidInputError(errors)

    features = compute_features(target, score, balls_bowled, wickets_lost)
    x = np.array([[float(features[c]) for c in FEATURE_COLUMNS]], dtype=np.float32)
    probabilities = session.run([meta["output"]["name"]], {meta["input"]["name"]: x})[0]
    return float(probabilities[0, meta["output"]["win_probability_column"]])


# ---------- Command line ----------

def _int_or_raw(text):
    # Leave non-integers as text so validation reports them with the contract's
    # message instead of argparse's generic "invalid int value".
    try:
        return int(text)
    except ValueError:
        return text


def main(argv=None):
    parser = argparse.ArgumentParser(description="Win probability for the chasing team in an IPL second innings.")
    parser.add_argument("--target", type=_int_or_raw, required=True, help="runs needed to win (1-300)")
    parser.add_argument("--score", type=_int_or_raw, required=True, help="runs scored so far (0 to target-1)")
    parser.add_argument("--balls", type=_int_or_raw, required=True, help="legal balls bowled so far (0-119)")
    parser.add_argument("--wickets", type=_int_or_raw, required=True, help="wickets lost so far (0-9)")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)

    try:
        p = predict_win_probability(args.target, args.score, args.balls, args.wickets)
    except InvalidInputError as e:
        for message in e.errors:
            print(f"error: {message}", file=sys.stderr)
        return INVALID_INPUT_EXIT_CODE

    f = compute_features(args.target, args.score, args.balls, args.wickets)
    version = default_model().meta["model_version"]
    if args.json:
        print(json.dumps({
            "winProbability": p,
            "modelVersion": version,
            "features": {"runsLeft": int(f["runs_left"]), "ballsLeft": int(f["balls_left"]),
                         "wicketsLeft": int(f["wickets_left"]), "crr": float(f["crr"]), "rrr": float(f["rrr"])},
        }))
    else:
        print(f"Chasing {args.target}: {args.score}/{args.wickets} after {args.balls} balls -> "
              f"need {int(f['runs_left'])} off {int(f['balls_left'])}, {int(f['wickets_left'])} wickets left")
        print(f"Win probability: {p:.2%}  (model {version})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
