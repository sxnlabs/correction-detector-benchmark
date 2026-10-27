"""Frozen snapshot of the chat log and the production scope applied to it."""

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from cdbench.detectors.regex import IGNORE_PATTERNS, normalise

PRIOR_REPLY_WINDOW_SECONDS = 30 * 60
PRIOR_REPLY_EXCERPT_CHARS = 500  # production keeps the first 500 characters of the reply
PRIOR_REPLY_CHARS = 300  # and the Haiku prompt keeps the last 300 of those
MAX_TEXT_CHARS = 2000
LLM_MIN_CHARS = 6

SNAPSHOT = "snapshot.jsonl"
SNAPSHOT_META = "snapshot.meta.json"


@dataclass(frozen=True)
class Message:
    id: str
    ts: datetime
    text: str
    prior_reply: str | None
    llm_eligible: bool


def snapshot(chat_log: Path, data_dir: Path) -> Path:
    """Copy the chat log once into `data_dir`; refuse to overwrite a snapshot."""
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / SNAPSHOT
    if target.exists():
        raise FileExistsError(f"{target} already exists")
    shutil.copyfile(chat_log, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    lines = sum(1 for _ in target.open(encoding="utf-8"))
    (data_dir / SNAPSHOT_META).write_text(
        json.dumps({"source": str(chat_log), "sha256": digest, "lines": lines,
                    "taken_at": datetime.now().astimezone().isoformat()}, indent=2))
    return target


def _llm_eligible(text: str) -> bool:
    norm = normalise(text).strip()
    return len(norm) >= LLM_MIN_CHARS and not any(p.search(norm) for p in IGNORE_PATTERNS)


def load_messages(data_dir: Path) -> list[Message]:
    """Messages production would classify, in log order."""
    messages: list[Message] = []
    previous: dict | None = None
    with (data_dir / SNAPSHOT).open(encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            entry = json.loads(raw)
            ts = datetime.fromisoformat(entry["ts"])
            text = (entry.get("user_msg") or "").strip()
            prior = None
            if previous is not None:
                age = (ts - datetime.fromisoformat(previous["ts"])).total_seconds()
                reply = previous.get("assistant_reply") or ""
                if 0 <= age <= PRIOR_REPLY_WINDOW_SECONDS and reply:
                    prior = reply[:PRIOR_REPLY_EXCERPT_CHARS][-PRIOR_REPLY_CHARS:]
            previous = entry
            if not text or text.startswith("/"):
                continue
            update_id = entry.get("tg_update_id")
            messages.append(Message(
                id=str(update_id) if update_id is not None else entry["ts"],
                ts=ts,
                text=text[:MAX_TEXT_CHARS],
                prior_reply=prior,
                llm_eligible=_llm_eligible(text),
            ))
    return messages
