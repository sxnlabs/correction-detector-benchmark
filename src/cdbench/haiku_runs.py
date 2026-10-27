"""Repeated Haiku runs on every eligible message, and the CLI prompt overhead."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from cdbench.claude import run_claude
from cdbench.dataset import Message
from cdbench.detectors.haiku import haiku_label, haiku_predict

OVERHEAD_PROMPT = "Réponds par un seul mot : other."
OVERHEAD_PROMPT_TOKENS = 10  # rough size of OVERHEAD_PROMPT itself


@dataclass
class HaikuRun:
    msg_id: str
    variant: str
    run: int
    label: bool | None
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_s: float
    error: str | None
    off_format: bool = False  # answer neither "correction" nor "other": production reads "other"


def run_haiku(messages: list[Message], *, runs: int, variants, cache_dir: Path,
              workers: int = 4) -> list[HaikuRun]:
    jobs = [(m, v, r) for m in messages if m.llm_eligible for v in variants for r in range(runs)]

    def one(job) -> HaikuRun:
        msg, variant, run = job
        call = haiku_predict(msg, variant=variant, run=run, cache_dir=cache_dir)
        text = (call.text or "").strip().lower()
        off_format = call.error is None and not text.startswith(("correction", "other"))
        return HaikuRun(msg.id, variant, run, haiku_label(call), call.model, call.input_tokens,
                        call.output_tokens, call.cost_usd, call.latency_s, call.error, off_format)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, jobs))


def majority(labels: list[bool | None]) -> bool | None:
    votes = Counter(label for label in labels if label is not None)
    if not votes:
        return None
    return votes[True] > votes[False]


def measure_overhead(cache_dir: Path) -> int:
    """Input tokens `claude -p` adds around a prompt (its own system prompt)."""
    call = run_claude(OVERHEAD_PROMPT, model="haiku", key="baseline|overhead",
                      cache_dir=cache_dir, timeout=60)
    if call.error is not None:
        raise RuntimeError(call.error)
    return call.input_tokens - OVERHEAD_PROMPT_TOKENS
