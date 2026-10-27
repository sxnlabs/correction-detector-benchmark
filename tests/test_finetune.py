from pathlib import Path

import pytest

from cdbench.detectors.finetune import cv_predict_finetune, inference_latency
from tests.synthetic import dataset

# hf-internal-testing/tiny-random-RobertaForSequenceClassification, downloaded once with
# curl (see README). CamemBERT shares the RoBERTa architecture.
TINY = str(Path.home() / ".cache/cdbench/models/tiny-roberta")
pytestmark = pytest.mark.skipif(not Path(TINY).exists(), reason="tiny model not downloaded")


@pytest.mark.slow
def test_cv_finetune_smoke():
    messages, gold, folds = dataset(5)
    preds = cv_predict_finetune(messages, gold, folds, variant="message", model_name=TINY,
                                epochs=1, lr=1e-3, device="cpu")
    assert set(preds) == {g.msg_id for g in gold}
    assert all(isinstance(label, bool) and 0.0 <= p <= 1.0 for label, p in preds.values())


@pytest.mark.slow
def test_inference_latency_smoke():
    messages, gold, _ = dataset(5)
    times = inference_latency(messages, gold, variant="with_prior", model_name=TINY,
                              epochs=1, lr=1e-3, device="cpu")
    assert len(times) == len(gold)
