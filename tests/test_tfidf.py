from cdbench.detectors.tfidf import cv_predict_tfidf, inference_latency
from tests.synthetic import dataset


def test_cv_predicts_every_message_once():
    messages, gold, folds = dataset()
    preds = cv_predict_tfidf(messages, gold, folds, variant="message")
    assert set(preds) == {g.msg_id for g in gold}
    assert all(0.0 <= p <= 1.0 for _, p in preds.values())
    accuracy = sum(preds[g.msg_id][0] == g.label for g in gold) / len(gold)
    assert accuracy > 0.8


def test_no_test_message_in_training():
    messages, gold, folds = dataset()
    seen = []
    cv_predict_tfidf(messages, gold, folds, variant="with_prior",
                     on_fit=lambda train_ids, test_ids: seen.append((train_ids, test_ids)))
    assert len(seen) == 5
    for train_ids, test_ids in seen:
        assert not set(train_ids) & set(test_ids)


def test_inference_latency_per_message():
    messages, gold, _ = dataset(6)
    times = inference_latency(messages, gold, variant="message")
    assert len(times) == len(messages) and all(t >= 0 for t in times)
