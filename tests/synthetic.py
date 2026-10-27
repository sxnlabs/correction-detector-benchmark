"""A small synthetic, separable dataset for model tests (no real messages)."""

from datetime import UTC, datetime

from cdbench.dataset import Message
from cdbench.folds import make_folds
from cdbench.labeling.arbitration import GoldLabel

POS = ["tu as encore oublie", "arrete de faire ca", "je t'ai deja dit", "c'est faux encore",
       "ne refais plus jamais ca", "tu te trompes encore"]
NEG = ["prepare le rapport", "quelle heure est il", "ajoute une tache", "merci beaucoup",
       "envoie le devis", "regarde la meteo"]


def dataset(n_each: int = 20):
    messages, gold = [], []
    for i in range(n_each):
        for kind, pool in (("p", POS), ("n", NEG)):
            mid = f"{kind}{i}"
            text = f"{pool[i % len(pool)]} {i}"
            messages.append(Message(mid, datetime(2026, 10, 1, tzinfo=UTC), text,
                                    "reponse precedente", True))
            gold.append(GoldLabel(mid, kind == "p", "opus"))
    return messages, gold, make_folds(gold)
