"""Checks on the committed ONNX artifacts. They run without retraining or the dataset."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as rt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features import FEATURE_COLUMNS, compute_features  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
META = json.loads((ARTIFACTS / "model_meta.json").read_text())
PARITY = json.loads((ARTIFACTS / "parity_cases.json").read_text())


@pytest.fixture(scope="module")
def session():
    return rt.InferenceSession(str(ARTIFACTS / "model.onnx"), providers=["CPUExecutionProvider"])


def predict(session, rows):
    x = np.asarray(rows, dtype=np.float32)
    return session.run([META["output"]["name"]], {META["input"]["name"]: x})[0][:, META["output"]["win_probability_column"]]


def test_meta_matches_model_file():
    digest = hashlib.sha256((ARTIFACTS / "model.onnx").read_bytes()).hexdigest()
    assert META["model_sha256"] == digest
    assert META["model_version"].endswith(digest[:12])


def test_meta_records_the_opsets_in_the_file():
    import onnx
    model = onnx.load(str(ARTIFACTS / "model.onnx"))
    in_file = {o.domain or "ai.onnx": o.version for o in model.opset_import}
    assert META["opset"] == in_file
    # Sorted, so exports are byte-identical across Python processes.
    assert [o.domain for o in model.opset_import] == sorted(o.domain for o in model.opset_import)


def test_input_and_output_names_and_shapes(session):
    (inp,), outputs = session.get_inputs(), {o.name: o for o in session.get_outputs()}
    assert inp.name == META["input"]["name"]
    assert inp.type == "tensor(float)"
    assert inp.shape[1] == len(FEATURE_COLUMNS) == len(META["input"]["feature_order"])
    assert META["input"]["feature_order"] == FEATURE_COLUMNS
    assert outputs[META["output"]["name"]].shape[1] == 2


def test_parity_cases_features_match_features_py():
    for case in PARITY["cases"]:
        i = case["input"]
        expected = compute_features(i["target"], i["score"], i["ballsBowled"], i["wicketsLost"])
        assert case["features"] == {c: float(expected[c]) for c in FEATURE_COLUMNS}, case["description"]


def test_parity_cases_probabilities_match_model(session):
    rows = [[case["features"][c] for c in FEATURE_COLUMNS] for case in PARITY["cases"]]
    expected = np.array([case["winProbability"] for case in PARITY["cases"]])
    assert np.abs(predict(session, rows) - expected).max() <= PARITY["tolerance"]


def test_parity_cases_respect_input_contract():
    contract = META["input_contract"]
    for case in PARITY["cases"]:
        i = case["input"]
        assert contract["target"]["min"] <= i["target"] <= contract["target"]["max"]
        assert contract["score"]["min"] <= i["score"] <= i["target"] - 1
        assert contract["ballsBowled"]["min"] <= i["ballsBowled"] <= contract["ballsBowled"]["max"]
        assert contract["wicketsLost"]["min"] <= i["wicketsLost"] <= contract["wicketsLost"]["max"]


def test_contract_has_specific_error_messages():
    messages = {field: rule["message"] for field, rule in META["input_contract"].items()}
    assert messages == {
        "target": "target must be between 1 and 300",
        "score": "score must be between 0 and target - 1",
        "ballsBowled": "ballsBowled must be between 0 and 119",
        "wicketsLost": "wicketsLost must be between 0 and 9",
    }


def test_parity_cases_include_required_edge_cases():
    descriptions = [case["description"] for case in PARITY["cases"]]
    assert "before the first ball" in descriptions
    assert any(d.startswith("effectively impossible") for d in descriptions)
    assert len(PARITY["cases"]) == 200


def test_matches_sklearn_when_trained_model_is_available(session):
    model_path = Path(__file__).resolve().parents[1] / "build" / "model.joblib"
    if not model_path.exists():
        pytest.skip("ml/build/model.joblib not built; run ml/evaluate.py")
    import joblib
    model = joblib.load(model_path)
    rows = np.array([[case["features"][c] for c in FEATURE_COLUMNS] for case in PARITY["cases"]], dtype=np.float32)
    assert np.abs(model.predict_proba(rows)[:, 1] - predict(session, rows)).max() <= PARITY["tolerance"]
