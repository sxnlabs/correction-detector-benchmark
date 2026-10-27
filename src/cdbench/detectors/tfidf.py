"""TF-IDF + logistic regression: the simplest machine-learning baseline."""

import time
from collections.abc import Callable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from cdbench.dataset import Message
from cdbench.detectors.inputs import Variant, model_input
from cdbench.labeling.arbitration import SEED, GoldLabel

C_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]


def _model() -> GridSearchCV:
    features = FeatureUnion([
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
        ("words", TfidfVectorizer(analyzer="word", ngram_range=(1, 1), sublinear_tf=True)),
    ])
    pipe = Pipeline([("features", features),
                     ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    return GridSearchCV(pipe, {"clf__C": C_GRID}, cv=inner, scoring="f1")


def cv_predict_tfidf(messages: list[Message], gold: list[GoldLabel], folds: list[list[str]], *,
                     variant: Variant,
                     on_fit: Callable[[list[str], list[str]], None] | None = None
                     ) -> dict[str, tuple[bool, float]]:
    """Out-of-fold label and probability for every gold message; C tuned on training folds."""
    by_id = {m.id: m for m in messages}
    labels = {g.msg_id: g.label for g in gold}
    preds: dict[str, tuple[bool, float]] = {}
    for test_ids in folds:
        test_set = set(test_ids)
        train_ids = [g.msg_id for g in gold if g.msg_id not in test_set]
        if on_fit:
            on_fit(train_ids, test_ids)
        model = _model()
        model.fit([model_input(by_id[i], variant) for i in train_ids],
                  [labels[i] for i in train_ids])
        proba = model.predict_proba([model_input(by_id[i], variant) for i in test_ids])[:, 1]
        for msg_id, p in zip(test_ids, proba, strict=True):
            preds[msg_id] = (bool(p >= 0.5), float(p))
    return preds


def inference_latency(messages: list[Message], gold: list[GoldLabel], *,
                      variant: Variant) -> list[float]:
    """Seconds to classify each message one at a time, model trained on all gold (CPU)."""
    by_id = {m.id: m for m in messages}
    model = _model()
    model.fit([model_input(by_id[g.msg_id], variant) for g in gold], [g.label for g in gold])
    times = []
    for g in gold:
        start = time.perf_counter()
        model.predict_proba([model_input(by_id[g.msg_id], variant)])
        times.append(time.perf_counter() - start)
    return times
