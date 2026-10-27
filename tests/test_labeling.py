from datetime import UTC, datetime

from cdbench.dataset import Message
from cdbench.labeling.arbitration import merge_gold, select_for_review
from cdbench.labeling.opus import OpusLabel, label_with_opus, opus_prompt, parse_opus_answer


def ol(i, label, conf=0.95):
    return OpusLabel(msg_id=str(i), label=label, confidence=conf, rationale="r")


def test_selection_priority_and_cap():
    opus, regex, haiku = [], {}, {}
    for i in range(30):  # opus vs haiku
        opus.append(ol(i, True)); regex[str(i)] = True; haiku[str(i)] = False
    for i in range(30, 50):  # low confidence, everyone agrees otherwise
        opus.append(ol(i, False, conf=0.5)); regex[str(i)] = False; haiku[str(i)] = False
    for i in range(50, 70):  # opus vs regex only
        opus.append(ol(i, True)); regex[str(i)] = False; haiku[str(i)] = True
    for i in range(70, 200):  # full agreement
        opus.append(ol(i, False)); regex[str(i)] = False; haiku[str(i)] = False
    items = select_for_review(opus, regex, haiku, cap=60, random_agreed=20)
    reasons = [it.reason for it in items]
    assert reasons[:30] == ["opus_vs_haiku"] * 30
    assert reasons[30:50] == ["low_confidence"] * 20
    assert reasons[50:60] == ["opus_vs_regex"] * 10
    assert reasons[60:] == ["random_agreed"] * 20


def test_random_agreed_sample_is_seeded_and_disjoint():
    opus = [ol(i, False) for i in range(100)] + [ol(100, True)]
    regex = {str(i): False for i in range(101)}
    haiku = {str(i): False for i in range(101)}
    first = select_for_review(opus, regex, haiku)
    second = select_for_review(opus, regex, haiku)
    assert first == second
    sampled = [it.msg_id for it in first if it.reason == "random_agreed"]
    assert len(sampled) == 20 and "100" not in sampled


def test_haiku_failure_is_not_a_disagreement():
    items = select_for_review([ol(1, True)], {"1": True}, {"1": None}, random_agreed=0)
    assert items == []


def test_unanswered_items_default_to_opus():
    gold = merge_gold([ol(1, True), ol(2, False)], answers={}, reviewed={"1"})
    assert {g.msg_id: g.resolved_by for g in gold} == {"1": "opus_default", "2": "opus"}


def test_nathan_answer_overrides_opus():
    gold = merge_gold([ol(1, True)], answers={"1": False}, reviewed={"1"})
    assert gold[0].label is False and gold[0].resolved_by == "nathan"


def test_parse_opus_answer_reads_embedded_json():
    label, conf, why = parse_opus_answer('Voici : {"label": "correction", "confidence": 0.8, '
                                         '"rationale": "reproche"}')
    assert (label, conf, why) == (True, 0.8, "reproche")


def test_opus_prompt_never_mentions_other_detectors():
    m = Message(id="1", ts=datetime(2026, 10, 1, tzinfo=UTC), text="t", prior_reply="p",
                llm_eligible=True)
    prompt = opus_prompt(m)
    assert "regex" not in prompt.lower() and "haiku" not in prompt.lower()
    assert "p" in prompt


def test_label_with_opus_skips_failed_calls(fake_claude, tmp_path):
    fake_claude.set(answer='{"label": "other", "confidence": 0.9, "rationale": "x"}')
    m = Message(id="1", ts=datetime(2026, 10, 1, tzinfo=UTC), text="t", prior_reply=None,
                llm_eligible=True)
    labels, failures = label_with_opus([m], cache_dir=tmp_path, workers=1)
    assert labels[0].label is False and failures == []


def test_unparseable_answer_is_retried_with_strict_prompt(fake_claude, tmp_path):
    fake_claude.set(answer="Je pense que c'est une correction.")
    m = Message(id="1", ts=datetime(2026, 10, 1, tzinfo=UTC), text="t", prior_reply=None,
                llm_eligible=True)
    labels, failures = label_with_opus([m], cache_dir=tmp_path, workers=1)
    assert labels == [] and failures == ["1"]
    assert fake_claude.calls == 2  # first answer plus one strict retry
    fake_claude.set(answer='{"label": "correction", "confidence": 0.6, "rationale": "x"}')
    labels, failures = label_with_opus([m], cache_dir=tmp_path, workers=1)
    assert labels[0].label is True and failures == []


def test_disagreement_counts_before_cap():
    from cdbench.labeling.arbitration import disagreement_counts

    opus = [ol(1, True), ol(2, False, conf=0.5), ol(3, True), ol(4, False)]
    regex = {"1": True, "2": False, "3": False, "4": False}
    haiku = {"1": False, "2": False, "3": True, "4": False}
    assert disagreement_counts(opus, regex, haiku) == {
        "opus_vs_haiku": 1, "low_confidence": 1, "opus_vs_regex": 1}
