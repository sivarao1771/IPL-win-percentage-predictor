"""Model definitions shared by train.py and evaluate.py.

Every model takes a DataFrame of chase states and returns P(chasing team wins).
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORY_COLUMNS, FEATURE_COLUMNS

SEED = 42
BUILD_DIR = Path(__file__).resolve().parent / "build"

BASELINE_COLUMNS = ["runs_left", "balls_left", "wickets_left"]

# Cricket logic, per feature in FEATURE_COLUMNS order. -1: a higher value can
# only lower the win probability, +1: only raise it, 0: no constraint.
MONOTONIC = {"runs_left": -1, "balls_left": 1, "wickets_left": 1, "runs_target": 0, "crr": 0, "rrr": -1}

# Small fixed grid for the main model, all compared on the validation season.
GRID = [
    {"learning_rate": lr, "max_iter": n, "max_leaf_nodes": leaves, "min_samples_leaf": 100, "monotonic": mono}
    for lr, n, leaves, mono in product([0.02, 0.05, 0.1], [25, 50, 100, 300], [7, 15, 31], [False, True])
]
# The ordered settings of the grid. A best value at either end of one of these
# means the search should have gone further in that direction.
GRID_AXES = {"learning_rate": [0.02, 0.05, 0.1], "max_iter": [25, 50, 100, 300], "max_leaf_nodes": [7, 15, 31]}


def load_split(name):
    return pd.read_csv(BUILD_DIR / f"{name}.csv")


def numeric(df, columns):
    # float32 is what the ONNX model will receive, so train on the same precision.
    return df[columns].to_numpy(dtype=np.float32)


class Model:
    """A named estimator plus the columns it reads."""

    def __init__(self, name, estimator, columns, uses_categories=False):
        self.name, self.estimator, self.columns = name, estimator, columns
        self.uses_categories = uses_categories

    def _inputs(self, df):
        return df[self.columns] if self.uses_categories else numeric(df, self.columns)

    def fit(self, df):
        self.estimator.fit(self._inputs(df), df["label"].to_numpy())
        return self

    def predict(self, df):
        return self.estimator.predict_proba(self._inputs(df))[:, 1]


def constant_baseline():
    """B0: always predicts the share of training rows where the chase was won."""
    return Model("B0 constant", DummyClassifier(strategy="prior"), BASELINE_COLUMNS)


def logistic_baseline():
    """B1: logistic regression on runs left, balls left and wickets left."""
    estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return Model("B1 logistic", estimator, BASELINE_COLUMNS)


def boosted_trees(params, name="M1 boosted trees"):
    """M1: gradient-boosted trees on the six match-state features."""
    monotonic = [MONOTONIC[c] for c in FEATURE_COLUMNS] if params["monotonic"] else None
    estimator = _hgb(params, monotonic)
    return Model(name, estimator, FEATURE_COLUMNS)


def boosted_trees_with_teams(params, train_df, name="M2 boosted trees + team/city"):
    """M2: M1 plus one-hot batting team, bowling team and city. Unseen categories
    (a new team or venue) are encoded as all zeros instead of failing."""
    encoder = ColumnTransformer([
        ("state", "passthrough", FEATURE_COLUMNS),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORY_COLUMNS),
    ])
    monotonic = None
    if params["monotonic"]:
        n_category_columns = sum(train_df[c].nunique() for c in CATEGORY_COLUMNS)
        monotonic = [MONOTONIC[c] for c in FEATURE_COLUMNS] + [0] * n_category_columns
    estimator = make_pipeline(encoder, _hgb(params, monotonic))
    return Model(name, estimator, FEATURE_COLUMNS + CATEGORY_COLUMNS, uses_categories=True)


def _hgb(params, monotonic):
    # No early stopping: it would hold out random *rows*, which leaks within a
    # match. The number of trees is chosen by the grid instead.
    return HistGradientBoostingClassifier(
        learning_rate=params["learning_rate"],
        max_iter=params["max_iter"],
        max_leaf_nodes=params["max_leaf_nodes"],
        min_samples_leaf=params["min_samples_leaf"],
        monotonic_cst=monotonic,
        early_stopping=False,
        random_state=SEED,
    )


def build_final(selection, train_df):
    params = selection["chosen"]["params"]
    if selection["chosen"]["family"] == "M2":
        return boosted_trees_with_teams(params, train_df, name="Final model (M2)")
    return boosted_trees(params, name="Final model (M1)")
