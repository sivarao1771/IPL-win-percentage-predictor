import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predict import (  # noqa: E402
    ARTIFACTS_DIR, INVALID_INPUT_EXIT_CODE, InvalidInputError, default_model, load_model,
    predict_win_probability,
)

ROOT = Path(__file__).resolve().parents[2]
PARITY = json.loads((ARTIFACTS_DIR / "parity_cases.json").read_text())
CASES = {case["description"]: case for case in PARITY["cases"]}
VALID = {"target": 180, "score": 95, "balls_bowled": 66, "wickets_lost": 3}


def predict(**overrides):
    return predict_win_probability(**{**VALID, **overrides})


def errors_for(**overrides):
    with pytest.raises(InvalidInputError) as info:
        predict(**overrides)
    return info.value.errors


# ---------- Matches the exported model ----------

def test_all_parity_cases_match_the_export():
    for case in PARITY["cases"]:
        i = case["input"]
        p = predict_win_probability(i["target"], i["score"], i["ballsBowled"], i["wicketsLost"])
        assert abs(p - case["winProbability"]) <= PARITY["tolerance"], case["description"]


@pytest.mark.parametrize("description", [
    "before the first ball",
    "one ball left",
    "one wicket left",
    "one run needed",
    "effectively impossible: 100 needed off 6 balls",
    "highest allowed target, before the first ball",
    "lowest allowed target",
    "everything at its maximum",
])
def test_edge_cases(description):
    i = CASES[description]["input"]
    p = predict_win_probability(i["target"], i["score"], i["ballsBowled"], i["wicketsLost"])
    assert 0 <= p <= 1
    assert p == pytest.approx(CASES[description]["winProbability"], abs=PARITY["tolerance"])


def test_returns_a_plain_unrounded_float():
    p = predict()
    assert type(p) is float
    assert p == pytest.approx(0.6356436014175415, abs=PARITY["tolerance"])


def test_numpy_integers_are_accepted():
    np = pytest.importorskip("numpy")
    assert predict(target=np.int64(180)) == predict()


# ---------- Valid boundaries ----------

@pytest.mark.parametrize("overrides", [
    {"target": 1, "score": 0},
    {"target": 300},
    {"score": 0},
    {"score": 179},
    {"balls_bowled": 0},
    {"balls_bowled": 119},
    {"wickets_lost": 0},
    {"wickets_lost": 9},
])
def test_boundaries_are_accepted(overrides):
    assert 0 <= predict(**overrides) <= 1


# ---------- Invalid input ----------

@pytest.mark.parametrize("overrides, expected", [
    ({"target": 0}, ["target must be between 1 and 300"]),
    ({"target": 301}, ["target must be between 1 and 300"]),
    ({"score": -1}, ["score must be between 0 and target - 1"]),
    ({"score": 180}, ["score must be between 0 and target - 1"]),
    ({"score": 250}, ["score must be between 0 and target - 1"]),
    ({"balls_bowled": -1}, ["ballsBowled must be between 0 and 119"]),
    ({"balls_bowled": 120}, ["ballsBowled must be between 0 and 119"]),
    ({"wickets_lost": -1}, ["wicketsLost must be between 0 and 9"]),
    ({"wickets_lost": 10}, ["wicketsLost must be between 0 and 9"]),
    ({"target": 180.0}, ["target must be an integer"]),
    ({"score": "95"}, ["score must be an integer"]),
    ({"balls_bowled": None}, ["ballsBowled must be an integer"]),
    ({"wickets_lost": True}, ["wicketsLost must be an integer"]),
])
def test_invalid_inputs_get_the_contract_message(overrides, expected):
    assert errors_for(**overrides) == expected


def test_every_problem_is_reported_at_once():
    assert errors_for(target=0, balls_bowled=120, wickets_lost="x") == [
        "target must be between 1 and 300",
        "ballsBowled must be between 0 and 119",
        "wicketsLost must be an integer",
    ]


@pytest.mark.parametrize("bad_target", [0, 301, "abc", None])
def test_invalid_target_does_not_also_flag_score(bad_target):
    errors = errors_for(target=bad_target, score=250)
    assert len(errors) == 1 and errors[0].startswith("target must be")


def test_range_messages_are_the_ones_in_model_meta():
    contract = default_model().meta["input_contract"]
    assert errors_for(target=0) == [contract["target"]["message"]]
    assert errors_for(score=-1) == [contract["score"]["message"]]
    assert errors_for(balls_bowled=120) == [contract["ballsBowled"]["message"]]
    assert errors_for(wickets_lost=10) == [contract["wicketsLost"]["message"]]


def test_invalid_input_error_is_a_value_error():
    with pytest.raises(ValueError, match="target must be between 1 and 300"):
        predict(target=0)


# ---------- Model loading ----------

def test_model_is_loaded_once():
    default_model.cache_clear()
    predict()
    predict(score=100)
    assert default_model.cache_info().misses == 1
    assert default_model.cache_info().hits >= 1


@pytest.fixture
def model_copy(tmp_path):
    for name in ("model.onnx", "model_meta.json"):
        shutil.copy(ARTIFACTS_DIR / name, tmp_path / name)
    return tmp_path


def edit_meta(model_dir, change):
    path = model_dir / "model_meta.json"
    meta = json.loads(path.read_text())
    change(meta)
    path.write_text(json.dumps(meta))


def test_load_model_accepts_the_real_files(model_copy):
    session, meta = load_model(model_copy)
    assert meta["model_version"] == default_model().meta["model_version"]
    assert predict_win_probability(180, 95, 66, 3, model=(session, meta)) == predict()


def test_load_model_fails_when_a_file_is_missing(model_copy):
    (model_copy / "model.onnx").unlink()
    with pytest.raises(FileNotFoundError, match="Model file not found: .*model.onnx"):
        load_model(model_copy)


def test_load_model_fails_on_sha256_mismatch(model_copy):
    edit_meta(model_copy, lambda m: m.update(model_sha256="0" * 64))
    with pytest.raises(ValueError, match="SHA-256 .* does not match"):
        load_model(model_copy)


def test_load_model_fails_on_wrong_feature_order(model_copy):
    edit_meta(model_copy, lambda m: m["input"].update(feature_order=m["input"]["feature_order"][::-1]))
    with pytest.raises(ValueError, match="Feature order"):
        load_model(model_copy)


def test_load_model_fails_on_wrong_input_name(model_copy):
    edit_meta(model_copy, lambda m: m["input"].update(name="wrong_input"))
    with pytest.raises(ValueError, match="'wrong_input' not found"):
        load_model(model_copy)


# ---------- Command line ----------

def run_cli(*args):
    return subprocess.run([sys.executable, str(ROOT / "ml" / "predict.py"), *args],
                          capture_output=True, text=True, cwd=ROOT)


def test_cli_prints_the_probability():
    result = run_cli("--target", "180", "--score", "95", "--balls", "66", "--wickets", "3")
    assert result.returncode == 0
    assert "need 85 off 54, 7 wickets left" in result.stdout
    assert "Win probability: 63.56%" in result.stdout


def test_cli_json_output():
    result = run_cli("--target", "180", "--score", "95", "--balls", "66", "--wickets", "3", "--json")
    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["winProbability"] == pytest.approx(predict(), abs=PARITY["tolerance"])
    assert body["modelVersion"] == default_model().meta["model_version"]
    assert body["features"] == {"runsLeft": 85, "ballsLeft": 54, "wicketsLeft": 7,
                                "crr": 95 * 6 / 66, "rrr": 85 * 6 / 54}


def test_cli_reports_invalid_input_with_exit_code_2():
    result = run_cli("--target", "301", "--score", "95.5", "--balls", "66", "--wickets", "10")
    assert result.returncode == INVALID_INPUT_EXIT_CODE
    assert result.stdout == ""
    # Compare only our error lines: on some machines onnxruntime itself prints a
    # hardware warning to stderr when it starts, which is not part of the CLI's output.
    errors = [line for line in result.stderr.splitlines() if line.startswith("error: ")]
    assert errors == [
        "error: target must be between 1 and 300",
        "error: score must be an integer",
        "error: wicketsLost must be between 0 and 9",
    ]
