"""First-pass ground truth: Opus labels every message, blind to the detectors."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from cdbench.claude import cached_call, run_claude
from cdbench.dataset import Message


@dataclass(frozen=True)
class OpusLabel:
    msg_id: str
    label: bool
    confidence: float
    rationale: str


DEFINITION = (
    "Une correction est un message où Nathan corrige son assistant : un reproche, "
    "une consigne durable (« ne fais plus X », « toujours Y », « désormais Z »), ou le "
    "signalement d'une erreur commise par l'assistant. Une demande, une question, une "
    "réponse courte ou une information neutre n'est pas une correction, même si elle "
    "commence par « non »."
)


def opus_prompt(msg: Message) -> str:
    context = f"\n\nRéponse précédente de l'assistant :\n{msg.prior_reply}" if msg.prior_reply else ""
    return (
        "Tu étiquettes des messages que Nathan envoie à son assistant Hermès sur Telegram, "
        "pour constituer une vérité terrain.\n\n"
        f"{DEFINITION}{context}\n\nMessage de Nathan :\n{msg.text}\n\n"
        'Réponds uniquement par un objet JSON : {"label": "correction" ou "other", '
        '"confidence": nombre entre 0 et 1, "rationale": une phrase courte}.'
    )


def parse_opus_answer(text: str) -> tuple[bool, float, str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in the answer")
    data = json.loads(match.group(0))
    label = str(data["label"]).strip().lower()
    if label not in ("correction", "other"):
        raise ValueError("unexpected label")
    confidence = float(data.get("confidence", 0.0))
    return label == "correction", max(0.0, min(1.0, confidence)), str(data.get("rationale", ""))


MAX_STRICT_RETRIES = 3
STRICT_SUFFIX = ("\n\nTa réponse précédente n'était pas un objet JSON. Réponds maintenant "
                 "par l'objet JSON seul, sans aucun autre texte.")


def _parse_for(msg_id: str, call) -> OpusLabel | None:
    if call.error is not None or call.text is None:
        return None
    try:
        label, confidence, rationale = parse_opus_answer(call.text)
    except (ValueError, KeyError, json.JSONDecodeError):
        return None
    return OpusLabel(msg_id, label, confidence, rationale)


def _label_one(msg: Message, cache_dir: Path) -> OpusLabel | str:
    base = run_claude(opus_prompt(msg), model="opus", key=f"opus|{msg.id}",
                      cache_dir=cache_dir, timeout=240)
    result = _parse_for(msg.id, base)
    if result is not None or base.error is not None:
        return result or msg.id
    # Unparseable answer: replay cached strict retries, then at most one new call per run.
    for attempt in range(MAX_STRICT_RETRIES):
        key = f"opus|{msg.id}|strict|{attempt}"
        fresh = cached_call(cache_dir, key) is None
        call = run_claude(opus_prompt(msg) + STRICT_SUFFIX, model="opus", key=key,
                          cache_dir=cache_dir, timeout=240)
        result = _parse_for(msg.id, call)
        if result is not None or fresh:
            return result or msg.id
    return msg.id


def label_with_opus(messages: list[Message], *, cache_dir: Path,
                    workers: int = 4) -> tuple[list[OpusLabel], list[str]]:
    """Label every message; returns the labels and the ids that could not be labeled."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda m: _label_one(m, cache_dir), messages))
    labels = [r for r in results if isinstance(r, OpusLabel)]
    failures = [r for r in results if isinstance(r, str)]
    return labels, failures
