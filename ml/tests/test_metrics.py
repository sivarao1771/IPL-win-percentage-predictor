import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import accuracy, bootstrap_ci, brier, by_phase, log_loss, reliability_table  # noqa: E402


def test_scores_by_hand():
    y = [1, 0, 1, 0]
    p = [0.9, 0.2, 0.4, 0.5]
    # 0.5 counts as predicting a chase win, so the last row is wrong.
    assert accuracy(y, p) == 0.5
    assert brier(y, p) == pytest.approx((0.01 + 0.04 + 0.36 + 0.25) / 4)
    expected = -(math.log(0.9) + math.log(0.8) + math.log(0.4) + math.log(0.5)) / 4
    assert log_loss(y, p) == pytest.approx(expected)


def test_coin_flip_reference_values():
    y = [0, 1] * 50
    assert log_loss(y, [0.5] * 100) == pytest.approx(math.log(2))
    assert brier(y, [0.5] * 100) == pytest.approx(0.25)


def test_log_loss_survives_certain_predictions():
    assert np.isfinite(log_loss([0, 1], [1.0, 0.0]))


def test_reliability_buckets():
    y = [1, 0, 1, 1]
    p = [0.05, 0.08, 0.95, 1.0]
    table = reliability_table(y, p)
    assert len(table) == 10
    assert table[0] == {"bucket": "0-10%", "rows": 2, "mean_predicted": pytest.approx(0.065),
                        "actual_win_rate": 0.5}
    # 1.0 lands in the top bucket.
    assert table[9]["rows"] == 2
    assert table[9]["actual_win_rate"] == 1.0
    assert table[5] == {"bucket": "50-60%", "rows": 0, "mean_predicted": None, "actual_win_rate": None}


def test_by_phase_uses_balls_bowled():
    result = by_phase([0, 36, 37, 119], [1, 1, 0, 0], [0.9, 0.9, 0.1, 0.9])
    assert result["overs 1-6"]["rows"] == 2
    assert result["overs 7-15"]["accuracy"] == 1.0
    assert result["overs 16-20"]["accuracy"] == 0.0


def test_bootstrap_resamples_whole_matches():
    # Match 1 is always predicted right, match 2 always wrong, with equal rows.
    # Any resample is made of whole matches, so accuracy is 0, 0.5 or 1 only.
    ids = [1, 1, 1, 2, 2, 2]
    y = [1, 1, 1, 0, 0, 0]
    p = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    low, high = bootstrap_ci(ids, y, p, n_resamples=200)["accuracy"]
    assert {low, high} <= {0.0, 0.5, 1.0}


def test_bootstrap_is_reproducible():
    rng = np.random.RandomState(0)
    ids = np.repeat(np.arange(30), 10)
    y = np.repeat(rng.randint(0, 2, 30), 10)
    p = rng.rand(300)
    assert bootstrap_ci(ids, y, p) == bootstrap_ci(ids, y, p)
