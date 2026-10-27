"""Classification scores, bootstrap intervals and latency percentiles."""

import numpy as np

from cdbench.labeling.arbitration import SEED


def scores(y_true: list[bool], y_pred: list[bool]) -> dict:
    t = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    tp, fp = int((t & p).sum()), int((~t & p).sum())
    fn, tn = int((t & ~p).sum()), int((~t & ~p).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def bootstrap_ci(y_true: list[bool], y_pred: list[bool], metric: str, n: int = 2000,
                 seed: int = SEED) -> tuple[float, float]:
    """95 % percentile interval of `metric` over `n` resamples of the messages."""
    rng = np.random.default_rng(seed)
    t = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(t), len(t))
        values.append(scores(t[idx].tolist(), p[idx].tolist())[metric])
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def paired_bootstrap_diff(y_true: list[bool], pred_a: list[bool], pred_b: list[bool],
                          metric: str, n: int = 2000,
                          seed: int = SEED) -> tuple[float, float, float]:
    """`metric(a) - metric(b)` and its 95 % interval, both detectors scored on the same
    resampled messages each time, so the shared difficulty of a message cancels out."""
    rng = np.random.default_rng(seed)
    t = np.asarray(y_true, dtype=bool)
    a = np.asarray(pred_a, dtype=bool)
    b = np.asarray(pred_b, dtype=bool)
    diff = scores(t.tolist(), a.tolist())[metric] - scores(t.tolist(), b.tolist())[metric]
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(t), len(t))
        values.append(scores(t[idx].tolist(), a[idx].tolist())[metric]
                      - scores(t[idx].tolist(), b[idx].tolist())[metric])
    return float(diff), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def latency_summary(seconds: list[float]) -> dict:
    arr = np.asarray(seconds, dtype=float)
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95))}


def wilson_interval(errors: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for a proportion (sound at 0 or n successes)."""
    if n == 0:
        return 0.0, 1.0
    p = errors / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
