"""Fail closed if any public output contains a real message or a slice of one."""

import re
from pathlib import Path

from cdbench.dataset import Message

MIN_WHOLE = 12  # shorter messages ("ok", "merci") cannot identify anyone


class PrivacyLeak(RuntimeError):
    pass


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _files(paths: list[Path]):
    for path in paths:
        path = Path(path)
        if path.is_dir():
            yield from (p for p in sorted(path.rglob("*")) if p.is_file())
        elif path.exists():
            yield path


def assert_no_message_text(paths: list[Path], messages: list[Message], window: int = 30) -> None:
    """Raise with file and message id only (never the matched text)."""
    contents = {f: _norm(f.read_bytes().decode("utf-8", errors="ignore")) for f in _files(paths)}
    for msg in messages:
        text = _norm(msg.text)
        if len(text) >= window:
            pieces = {text[i:i + window] for i in range(len(text) - window + 1)}
        elif len(text) >= MIN_WHOLE:
            pieces = {text}
        else:
            continue
        for f, content in contents.items():
            if any(piece in content for piece in pieces):
                raise PrivacyLeak(f"{f}: contains text from message {msg.id}")
