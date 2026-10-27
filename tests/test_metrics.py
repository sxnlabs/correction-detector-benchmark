import pytest

from cdbench.cost import cost_per_1000, direct_input_tokens
from cdbench.metrics import bootstrap_ci, latency_summary, scores


def matrix(tp, fp, fn, tn):
    y_true = [True] * tp + [False] * fp + [True] * fn + [False] * tn
    y_pred = [True] * tp + [True] * fp + [False] * fn + [False] * tn
    return y_true, y_pred


def test_scores_known_matrix():
    s = scores(*matrix(6, 2, 4, 8))
    assert s["precision"] == pytest.approx(0.75)
    assert s["recall"] == pytest.approx(0.6)
    assert s["f1"] == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (6, 2, 4, 8)


def test_scores_no_prediction_is_zero_not_error():
    s = scores([True, False], [False, False])
    assert s["precision"] == 0.0 and s["recall"] == 0.0 and s["f1"] == 0.0


def test_bootstrap_ci_contains_point_estimate():
    y_true, y_pred = matrix(30, 10, 20, 200)
    low, high = bootstrap_ci(y_true, y_pred, "recall")
    assert low <= 0.6 <= high and high - low < 0.5


def test_latency_summary():
    s = latency_summary([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s["p50"] == 3.0 and s["p95"] > 4.0


def test_cost_per_1000_formula():
    # 1000 messages of 150 input and 10 output tokens at $0.10 and $0.50 per MTok
    assert cost_per_1000([150] * 1000, [10] * 1000) == pytest.approx(0.015 + 0.005)


def test_direct_input_tokens_removes_cli_overhead():
    assert direct_input_tokens(3000, overhead=2850) == 150
    assert direct_input_tokens(100, overhead=2850) == 1


def test_wilson_interval_for_zero_errors_out_of_20():
    from cdbench.metrics import wilson_interval

    low, high = wilson_interval(0, 20)
    assert low == 0.0 and 0.15 < high < 0.18


def test_paired_bootstrap_diff_sign_and_interval():
    from cdbench.metrics import paired_bootstrap_diff

    y = [True] * 20 + [False] * 20
    perfect = list(y)
    noisy = [True] * 10 + [False] * 10 + [False] * 20  # half the positives missed
    diff, low, high = paired_bootstrap_diff(y, perfect, noisy, "f1")
    assert abs(diff - (1.0 - 2 / 3)) < 1e-9
    assert 0 < low <= diff <= high
    same = paired_bootstrap_diff(y, noisy, noisy, "f1")
    assert same == (0.0, 0.0, 0.0)  # identical predictions: no difference in any resample
