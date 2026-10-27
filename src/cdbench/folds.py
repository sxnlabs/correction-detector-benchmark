"""Stratified folds, fixed once per run."""

from sklearn.model_selection import StratifiedKFold

from cdbench.labeling.arbitration import SEED, GoldLabel


def make_folds(gold: list[GoldLabel], k: int = 5, seed: int = SEED) -> list[list[str]]:
    ids = [g.msg_id for g in gold]
    labels = [g.label for g in gold]
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [[ids[i] for i in test] for _, test in splitter.split(ids, labels)]
