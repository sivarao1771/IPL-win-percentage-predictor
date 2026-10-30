import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import paired_log_loss_diff_ci  # noqa: E402
from train import choose_setting  # noqa: E402


def setting(trees, leaves, monotonic, learning_rate, log_loss):
    return {"params": {"max_iter": trees, "max_leaf_nodes": leaves, "monotonic": monotonic,
                       "learning_rate": learning_rate},
            "scores": {"log_loss": log_loss}}


def test_settings_more_than_1_percent_worse_are_not_eligible():
    results = [setting(50, 31, True, 0.1, 0.3800), setting(25, 7, True, 0.1, 0.3900)]
    # 0.39 is 2.6% worse than 0.38, so the smaller model is not eligible.
    assert choose_setting(results)["params"]["max_iter"] == 50


def test_monotonic_preferred_over_smaller():
    results = [setting(100, 7, False, 0.1, 0.3804), setting(50, 15, True, 0.1, 0.3829),
               setting(50, 31, True, 0.1, 0.3803)]
    # Mirrors the real 2024 grid: 100 x 7 is smaller but not monotonic.
    assert choose_setting(results)["params"] == {"max_iter": 50, "max_leaf_nodes": 15,
                                                 "monotonic": True, "learning_rate": 0.1}


def test_then_smallest_then_higher_learning_rate():
    results = [setting(300, 7, True, 0.02, 0.3832), setting(300, 7, True, 0.05, 0.3810),
               setting(100, 31, True, 0.05, 0.3810), setting(300, 15, True, 0.02, 0.3806)]
    # 300 x 7 = 2100 is smallest; the two 2100 settings are split by the higher learning rate.
    assert choose_setting(results)["params"] == {"max_iter": 300, "max_leaf_nodes": 7,
                                                 "monotonic": True, "learning_rate": 0.05}


def test_paired_difference_is_zero_for_identical_models():
    ids, y, p = [1, 1, 2, 2], [1, 1, 0, 0], [0.7, 0.6, 0.4, 0.2]
    assert paired_log_loss_diff_ci(ids, y, p, p) == [0.0, 0.0]


def test_paired_difference_flags_a_clearly_worse_model():
    rng = np.random.RandomState(0)
    ids = np.repeat(np.arange(40), 5)
    y = np.repeat(rng.randint(0, 2, 40), 5)
    good = np.where(y == 1, 0.8, 0.2)
    bad = np.full(len(y), 0.5)
    low, high = paired_log_loss_diff_ci(ids, y, good, bad)
    assert low > 0  # the coin-flip model is worse on every resample
