from cdbench.folds import make_folds
from cdbench.labeling.arbitration import GoldLabel


def gold(n_pos, n_neg):
    return ([GoldLabel(f"p{i}", True, "opus") for i in range(n_pos)]
            + [GoldLabel(f"n{i}", False, "opus") for i in range(n_neg)])


def test_folds_stratified_within_one_positive():
    folds = make_folds(gold(23, 237))
    positives = [sum(i.startswith("p") for i in f) for f in folds]
    assert len(folds) == 5
    assert max(positives) - min(positives) <= 1
    assert sorted(i for f in folds for i in f) == sorted(g.msg_id for g in gold(23, 237))


def test_folds_stable_for_seed():
    assert make_folds(gold(10, 40)) == make_folds(gold(10, 40))
    assert make_folds(gold(10, 40)) != make_folds(gold(10, 40), seed=1)
