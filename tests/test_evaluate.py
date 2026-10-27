import json

import pytest

from cdbench.evaluate import evaluate
from cdbench.haiku_runs import HaikuRun
from cdbench.labeling.arbitration import ReviewItem
from cdbench.labeling.opus import OpusLabel
from cdbench.store import write_rows
from tests.conftest import line
from tests.synthetic import NEG, POS


def test_evaluate_without_finetune(write_log, tmp_path):
    from cdbench.dataset import snapshot

    entries, gold, opus, runs = [], [], [], []
    for i in range(40):
        pos = i % 2 == 0
        text = f"{(POS if pos else NEG)[i % 6]} {i}"
        entries.append(line(i * 60, text, update_id=i))
        gold.append({"msg_id": str(i), "label": pos, "resolved_by": "opus"})
        opus.append(OpusLabel(str(i), pos, 0.9, "r"))
        for variant in ("message", "with_prior"):
            for r in range(3):
                runs.append(HaikuRun(str(i), variant, r, pos, "claude-haiku-5-5", 3000, 20,
                                     0.0007, 4.0, None))
    data = tmp_path / "data"
    snapshot(write_log(entries), data)
    (data / "gold.jsonl").write_text("\n".join(json.dumps(g) for g in gold) + "\n")
    write_rows(data / "opus.jsonl", opus)
    write_rows(data / "haiku_runs.jsonl", runs)
    write_rows(data / "review.jsonl", [ReviewItem("0", "random_agreed")])
    (data / "answers.json").write_text(json.dumps({"0": False}))
    (data / "haiku_overhead.json").write_text('{"overhead_input_tokens": 2850}')

    summary, _meta, probas, _gold = evaluate(data, finetune=False)
    assert {"regex", "haiku", "haiku+prior", "pipeline", "tfidf", "tfidf+prior"} <= set(
        summary["detectors"])
    assert summary["detectors"]["haiku"]["recall"] == 1.0
    assert summary["cost_direct_per_1000"]["haiku"] == (150 * 0.10 + 20 * 0.50) / 1000
    assert summary["opus_error_rate_random_sample"] == 1.0
    assert summary["counts"] == {"in_scope": 40, "positives": 20, "llm_eligible": 40}
    assert set(probas) == {"tfidf", "tfidf+prior"}
    assert (data / "predictions.json").exists()


def test_pipeline_cost_counts_only_messages_sent_to_haiku():
    from cdbench.evaluate import pipeline_cost_and_latency

    regex_hits = {"a": True, "b": False, "c": False, "d": False}
    eligible = {"a", "b", "c"}  # d is too short for Haiku
    cost, latency = pipeline_cost_and_latency(
        regex_hits, eligible, haiku_cost_per_1000=0.08,
        regex_latencies=[0.001] * 4, haiku_latencies={"b": 2.0, "c": 4.0})
    assert cost == 0.08 * 2 / 4  # b and c reach Haiku
    assert latency["p50"] == pytest.approx(1.001)  # regex time, plus Haiku time for b and c


def test_haiku_headline_is_one_call_and_vote_is_separate():
    from cdbench.evaluate import _haiku

    def run(msg_id, r, label):
        return HaikuRun(msg_id, "message", r, label, "claude-haiku-5-5", 3000, 20, 0.0006,
                        2.0, None)

    runs = [run("a", 0, False), run("a", 1, True), run("a", 2, True)]
    preds, _spread, _cli, direct, _lat, _per_call = _haiku(
        runs, ["a", "b"], {"a"}, {"a": True, "b": False}, overhead=2850)
    assert preds["haiku"]["a"] is False  # run 0, like the single production call
    assert preds["haiku vote3"]["a"] is True
    per_call = (150 * 0.10 + 20 * 0.50) / 1_000_000
    assert direct["haiku"] == pytest.approx(1000 * per_call * 1 / 2)  # only a is eligible
    assert direct["haiku vote3"] == pytest.approx(3 * direct["haiku"])


def test_folds_rebuilt_when_gold_changes(tmp_path):
    from cdbench.evaluate import _folds
    from cdbench.labeling.arbitration import GoldLabel

    gold = [GoldLabel(str(i), i % 4 == 0, "opus") for i in range(40)]
    first = _folds(tmp_path, gold)
    assert _folds(tmp_path, gold) == first
    flipped = [GoldLabel(g.msg_id, (not g.label) if g.msg_id == "1" else g.label, "nathan")
               for g in gold]
    rebuilt = _folds(tmp_path, flipped)
    positives = [sum(int(i) % 4 == 0 or i == "1" for i in f) for f in rebuilt]
    assert max(positives) - min(positives) <= 1
    assert json.loads((tmp_path / "folds.json").read_text())["gold_hash"]
