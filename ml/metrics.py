"""Evaluation metrics for win probabilities.

y is 1 if the chasing team won, p is the predicted probability of that.
"""
import numpy as np

from features import PHASES

EPS = 1e-15


def accuracy(y, p):
    """Share of rows where the side of 50% matches the result."""
    return float(np.mean((np.asarray(p) >= 0.5) == np.asarray(y)))


def log_loss(y, p):
    """Average penalty -log(probability given to what happened). Confident and
    wrong is punished hard. Lower is better; always guessing 0.5 scores 0.693."""
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    """Mean squared gap between probability and result. Lower is better;
    always guessing 0.5 scores 0.25."""
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y)) ** 2))


def scores(y, p):
    return {"rows": int(len(y)), "accuracy": accuracy(y, p), "log_loss": log_loss(y, p), "brier": brier(y, p)}


def by_phase(balls_bowled, y, p):
    balls_bowled, y, p = np.asarray(balls_bowled), np.asarray(y), np.asarray(p)
    return {name: scores(y[mask], p[mask])
            for name, low, high in PHASES
            for mask in [(balls_bowled >= low) & (balls_bowled <= high)]}


def reliability_table(y, p, n_buckets=10):
    """Group predictions into equal-width buckets and compare the average
    prediction with how often the chasing team actually won."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    # A prediction of exactly 1.0 belongs in the top bucket, not a new one.
    bucket = np.minimum((p * n_buckets).astype(int), n_buckets - 1)
    table = []
    for b in range(n_buckets):
        mask = bucket == b
        rows = int(mask.sum())
        table.append({
            "bucket": f"{b * 100 // n_buckets}-{(b + 1) * 100 // n_buckets}%",
            "rows": rows,
            "mean_predicted": float(p[mask].mean()) if rows else None,
            "actual_win_rate": float(y[mask].mean()) if rows else None,
        })
    return table


def _per_row_log_loss(y, p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_log_loss_diff_ci(match_ids, y, p_reference, p_other, n_resamples=1000, seed=42, level=0.95):
    """Confidence interval for log_loss(other) - log_loss(reference), resampling
    whole matches. Both models are scored on the same resampled matches, so luck
    in which matches were drawn cancels out. If the interval includes 0, the data
    can't tell the two models apart."""
    match_ids, y = np.asarray(match_ids), np.asarray(y)
    unique, inverse = np.unique(match_ids, return_inverse=True)
    counts = np.bincount(inverse)
    diff = np.bincount(inverse, weights=_per_row_log_loss(y, p_other) - _per_row_log_loss(y, p_reference))
    picks = np.random.RandomState(seed).randint(0, len(unique), size=(n_resamples, len(unique)))
    values = diff[picks].sum(axis=1) / counts[picks].sum(axis=1)
    tail = (1 - level) / 2 * 100
    return [float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))]


def bootstrap_ci(match_ids, y, p, n_resamples=1000, seed=42, level=0.95):
    """Confidence intervals by resampling whole matches with replacement.

    Rows from one match move together (same result, similar states), so
    resampling rows would make the intervals look narrower than they are.
    Each metric is a mean of a per-row quantity, so per-match sums are enough.
    """
    match_ids, y = np.asarray(match_ids), np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    per_row = {
        "accuracy": ((p >= 0.5) == y).astype(float),
        "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)),
        "brier": (p - y) ** 2,
    }
    unique, inverse = np.unique(match_ids, return_inverse=True)
    counts = np.bincount(inverse)
    sums = {name: np.bincount(inverse, weights=values) for name, values in per_row.items()}

    rng = np.random.RandomState(seed)
    picks = rng.randint(0, len(unique), size=(n_resamples, len(unique)))
    rows = counts[picks].sum(axis=1)
    tail = (1 - level) / 2 * 100
    intervals = {}
    for name, s in sums.items():
        values = s[picks].sum(axis=1) / rows
        intervals[name] = [float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))]
    return intervals
