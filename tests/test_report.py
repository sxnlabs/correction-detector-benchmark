import json

from cdbench.report import DETECTORS, render_table, summarize, write_report


def toy():
    gold = {"id-alpha": True, "id-beta": True, "id-gamma": False, "id-delta": False}
    preds = {name: {"id-alpha": True, "id-beta": False, "id-gamma": False, "id-delta": True}
             for name in DETECTORS}
    extra = {"cost_cli_per_1000": {}, "cost_direct_per_1000": {}, "latency": {},
             "haiku_run_spread": {}, "counts": {"in_scope": 4, "positives": 2},
             "resolved_by": {"opus": 4}, "opus_error_rate_random_sample": 0.05}
    return gold, preds, extra


def test_summary_contains_only_aggregates():
    gold, preds, extra = toy()
    summary = summarize(gold, preds, extra)
    allowed = {"precision", "recall", "f1", "tp", "fp", "fn", "tn", "precision_ci",
               "recall_ci", "f1_ci"}
    for name in DETECTORS:
        assert set(summary["detectors"][name]) == allowed
    assert "id-" not in json.dumps(summary["detectors"])  # no message ids in the scores


def test_table_lists_every_detector():
    gold, preds, extra = toy()
    table = render_table(summarize(gold, preds, extra))
    for name in DETECTORS:
        assert name in table


def test_write_report_runs_privacy_guard(tmp_path):
    gold, preds, extra = toy()
    write_report(summarize(gold, preds, extra), {"seed": 1}, tmp_path, messages=[],
                 extra_paths=[])
    assert (tmp_path / "summary.json").exists() and (tmp_path / "table.md").exists()


def test_summary_reports_paired_f1_differences():
    from cdbench.report import COMPARISONS

    gold, preds, extra = toy()
    summary = summarize(gold, preds, extra)
    diffs = summary["f1_differences"]
    assert set(diffs) == {f"{a} - {b}" for a, b in COMPARISONS}
    assert set(diffs["haiku - tfidf"]) == {"diff", "ci"}
    assert "F1 difference" in render_table(summary)


def test_pipeline_is_compared_with_the_haiku_variant_it_runs():
    # production calls Haiku with the previous reply, so only that pair isolates the regex
    from cdbench.report import COMPARISONS

    assert ("pipeline", "haiku+prior") in COMPARISONS
    assert ("pipeline", "haiku") not in COMPARISONS


def test_context_effect_on_trained_models_is_compared_paired():
    from cdbench.report import COMPARISONS

    assert ("tfidf", "tfidf+prior") in COMPARISONS
    assert ("finetune", "finetune+prior") in COMPARISONS
