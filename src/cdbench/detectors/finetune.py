"""A small French encoder fine-tuned with PyTorch, evaluated by cross-validation.

Fixed hyperparameters (no tuning on test folds): a fixed number of epochs, a
linear schedule with 10 % warmup, and a class-weighted loss for the imbalance.
"""

import random
import time

import numpy as np
import torch
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from cdbench.dataset import Message
from cdbench.detectors.inputs import Variant, model_input
from cdbench.labeling.arbitration import SEED, GoldLabel

DEFAULT_MODEL = "almanach/camembert-base"
MAX_LENGTH = 192
BATCH_SIZE = 16


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train(texts: list[str], labels: list[bool], *, model_name: str, epochs: int, lr: float,
           device: str, seed: int):
    _seed_everything(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    y = torch.tensor([int(v) for v in labels])
    positives = max(int(y.sum()), 1)
    weights = torch.tensor([len(y) / (2 * (len(y) - positives)), len(y) / (2 * positives)],
                           dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(texts) + BATCH_SIZE - 1) // BATCH_SIZE)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(texts), generator=generator).tolist()
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            batch = tokenizer([texts[i] for i in idx], truncation=True, max_length=MAX_LENGTH,
                              padding=True, return_tensors="pt").to(device)
            logits = model(**batch).logits
            loss = loss_fn(logits, y[idx].to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    model.eval()
    return tokenizer, model


@torch.no_grad()
def _proba(tokenizer, model, texts: list[str], device: str) -> list[float]:
    out = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = tokenizer(texts[start:start + BATCH_SIZE], truncation=True,
                          max_length=MAX_LENGTH, padding=True, return_tensors="pt").to(device)
        out.extend(torch.softmax(model(**batch).logits, dim=-1)[:, 1].cpu().tolist())
    return out


def cv_predict_finetune(messages: list[Message], gold: list[GoldLabel], folds: list[list[str]],
                        *, variant: Variant, model_name: str = DEFAULT_MODEL, epochs: int = 6,
                        lr: float = 3e-5, seed: int = SEED, device: str | None = None
                        ) -> dict[str, tuple[bool, float]]:
    """Out-of-fold label and probability for every gold message."""
    device = device or default_device()
    by_id = {m.id: m for m in messages}
    labels = {g.msg_id: g.label for g in gold}
    preds: dict[str, tuple[bool, float]] = {}
    for test_ids in folds:
        test_set = set(test_ids)
        train_ids = [g.msg_id for g in gold if g.msg_id not in test_set]
        tokenizer, model = _train([model_input(by_id[i], variant) for i in train_ids],
                                  [labels[i] for i in train_ids], model_name=model_name,
                                  epochs=epochs, lr=lr, device=device, seed=seed)
        probs = _proba(tokenizer, model, [model_input(by_id[i], variant) for i in test_ids],
                       device)
        for msg_id, p in zip(test_ids, probs, strict=True):
            preds[msg_id] = (bool(p >= 0.5), float(p))
    return preds


def inference_latency(messages: list[Message], gold: list[GoldLabel], *, variant: Variant,
                      model_name: str = DEFAULT_MODEL, epochs: int = 6, lr: float = 3e-5,
                      device: str | None = None, seed: int = SEED) -> list[float]:
    """Seconds to classify each message one at a time, model trained on all gold."""
    device = device or default_device()
    by_id = {m.id: m for m in messages}
    texts = [model_input(by_id[g.msg_id], variant) for g in gold]
    tokenizer, model = _train(texts, [g.label for g in gold], model_name=model_name,
                              epochs=epochs, lr=lr, device=device, seed=seed)
    _proba(tokenizer, model, texts[:1], device)  # warm-up
    times = []
    for text in texts:
        start = time.perf_counter()
        _proba(tokenizer, model, [text], device)
        if device == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - start)
    return times
