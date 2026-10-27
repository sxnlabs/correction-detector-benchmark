import json
import os
from datetime import UTC, datetime, timedelta

import pytest


@pytest.fixture
def write_log(tmp_path):
    """Write synthetic chat-log lines (never real messages) and return the path."""

    def _write(entries):
        path = tmp_path / "chat.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        return path

    return _write


def line(minutes: float, text: str, reply: str = "reply", update_id: int | None = None):
    ts = datetime(2026, 10, 1, 9, 0, tzinfo=UTC) + timedelta(minutes=minutes)
    return {"ts": ts.isoformat(), "user_msg": text, "assistant_reply": reply,
            "tg_update_id": update_id}


FAKE_CLAUDE = r'''#!/usr/bin/env python3
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "calls.log"), "a") as f:
    f.write("call\n")
cfg_path = os.path.join(here, "config.json")
cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
if cfg.get("fail"):
    print("quota exceeded", file=sys.stderr)
    sys.exit(1)
print(json.dumps({
    "type": "result", "is_error": False, "result": cfg.get("answer", "other"),
    "total_cost_usd": 0.001,
    "usage": {"input_tokens": 10, "cache_creation_input_tokens": 100,
              "cache_read_input_tokens": 0, "output_tokens": 5},
    "modelUsage": {"claude-haiku-5-5": {"costUSD": 0.001}},
}))
'''


class FakeClaude:
    def __init__(self, bin_dir):
        self.bin_dir = bin_dir

    def set(self, **cfg):
        (self.bin_dir / "config.json").write_text(json.dumps(cfg))

    @property
    def calls(self):
        log = self.bin_dir / "calls.log"
        return len(log.read_text().splitlines()) if log.exists() else 0


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    """Put a fake `claude` binary first on PATH, configured through a file."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "claude"
    exe.write_text(FAKE_CLAUDE)
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    return FakeClaude(bin_dir)
