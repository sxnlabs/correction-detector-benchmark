"""Which messages Nathan arbitrates, and the merged gold labels."""

import random
from dataclasses import dataclass
from typing import Literal

from cdbench.labeling.opus import OpusLabel

LOW_CONFIDENCE = 0.7
SEED = 20261026

Reason = Literal["opus_vs_haiku", "low_confidence", "opus_vs_regex", "random_agreed"]


@dataclass(frozen=True)
class ReviewItem:
    msg_id: str
    reason: Reason


@dataclass(frozen=True)
class GoldLabel:
    msg_id: str
    label: bool
    resolved_by: Literal["opus", "nathan", "opus_default"]


def _bucket(lab: OpusLabel, regex: dict[str, bool],
            haiku_first_run: dict[str, bool | None]) -> Reason | None:
    haiku = haiku_first_run.get(lab.msg_id)
    if haiku is not None and haiku != lab.label:
        return "opus_vs_haiku"
    if lab.confidence < LOW_CONFIDENCE:
        return "low_confidence"
    if regex.get(lab.msg_id) is not None and regex[lab.msg_id] != lab.label:
        return "opus_vs_regex"
    return None


def disagreement_counts(opus: list[OpusLabel], regex: dict[str, bool],
                        haiku_first_run: dict[str, bool | None]) -> dict[str, int]:
    """Size of each disagreement bucket before the cap."""
    counts = {"opus_vs_haiku": 0, "low_confidence": 0, "opus_vs_regex": 0}
    for lab in opus:
        reason = _bucket(lab, regex, haiku_first_run)
        if reason:
            counts[reason] += 1
    return counts


def select_for_review(opus: list[OpusLabel], regex: dict[str, bool],
                      haiku_first_run: dict[str, bool | None], *, cap: int = 60,
                      random_agreed: int = 20, seed: int = SEED) -> list[ReviewItem]:
    """Disagreements by priority (capped), then a seeded sample of agreed messages."""
    buckets: dict[Reason, list[str]] = {"opus_vs_haiku": [], "low_confidence": [],
                                        "opus_vs_regex": []}
    agreed: list[str] = []
    for lab in opus:
        reason = _bucket(lab, regex, haiku_first_run)
        if reason:
            buckets[reason].append(lab.msg_id)
        else:
            agreed.append(lab.msg_id)
    items: list[ReviewItem] = []
    for reason, ids in buckets.items():
        for msg_id in ids:
            if len(items) < cap:
                items.append(ReviewItem(msg_id, reason))
    sample = random.Random(seed).sample(sorted(agreed), min(random_agreed, len(agreed)))
    items.extend(ReviewItem(msg_id, "random_agreed") for msg_id in sample)
    return items


def merge_gold(opus: list[OpusLabel], answers: dict[str, bool],
               reviewed: set[str]) -> list[GoldLabel]:
    """Nathan's answer wins; a reviewed message he left unanswered keeps the Opus label."""
    gold = []
    for lab in opus:
        if lab.msg_id in answers:
            gold.append(GoldLabel(lab.msg_id, answers[lab.msg_id], "nathan"))
        elif lab.msg_id in reviewed:
            gold.append(GoldLabel(lab.msg_id, lab.label, "opus_default"))
        else:
            gold.append(GoldLabel(lab.msg_id, lab.label, "opus"))
    return gold
