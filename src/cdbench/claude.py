"""Cached `claude -p` runner, hardened like Hermes `lib/llm.py`.

Every call is appended to `cache_dir/calls.jsonl` under a key. A key with a
successful entry is never called again, so an interrupted run resumes where it
stopped. A failed call is recorded with its error and retried next time; it is
never turned into a label. Errors never echo the prompt (it holds private text).
"""

import fcntl
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ENV_PASSTHROUGH = ("PATH", "HOME", "LANG", "USER", "TMPDIR")
CALLS = "calls.jsonl"


@dataclass
class ClaudeCall:
    key: str
    model: str
    text: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_s: float
    error: str | None


def command(prompt: str, model: str, claude_bin: str) -> list[str]:
    """Production argv (Hermes lib/llm.py `command`) with JSON output."""
    return [claude_bin, "-p", "--model", model, "--output-format", "json",
            "--tools", "", "--no-session-persistence", "--restricted", "--safe-mode",
            "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
            "--permission-mode", "dontAsk", prompt]


def _environment() -> dict:
    env = {k: os.environ[k] for k in ENV_PASSTHROUGH if os.environ.get(k)}
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    return env


def cached_call(cache_dir: Path, key: str) -> ClaudeCall | None:
    path = cache_dir / CALLS
    if not path.exists():
        return None
    found = None
    with path.open(encoding="utf-8") as f:
        for raw in f:
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue  # a run killed mid-write leaves a truncated line
            if entry.get("key") == key and entry.get("error") is None:
                found = ClaudeCall(**entry)
    return found


def _append(cache_dir: Path, call: ClaudeCall) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / CALLS).open("a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(asdict(call), ensure_ascii=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def _parse(stdout: str) -> tuple[str, str, int, int, float | None]:
    data = json.loads(stdout)
    if data.get("is_error"):
        raise ValueError(f"claude reported an error ({data.get('subtype', 'unknown')})")
    usage = data.get("usage") or {}
    input_tokens = sum(usage.get(k, 0) for k in
                       ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    models = list((data.get("modelUsage") or {}).keys())
    return (data.get("result") or "", models[0] if models else "unknown",
            input_tokens, usage.get("output_tokens", 0), data.get("total_cost_usd"))


def run_claude(prompt: str, *, model: str, key: str, cache_dir: Path,
               timeout: int = 120) -> ClaudeCall:
    """Run one cached `claude -p` call; reuse a successful cached call for `key`."""
    cached = cached_call(cache_dir, key)
    if cached is not None:
        return cached
    claude_bin = shutil.which("claude") or "claude"
    start = time.monotonic()
    error = None
    text, served, tin, tout, cost = None, model, 0, 0, None
    try:
        with tempfile.TemporaryDirectory(prefix="cdbench-") as cwd:
            proc = subprocess.run(command(prompt, model, claude_bin), check=False, cwd=cwd,
                                  capture_output=True, text=True, timeout=timeout,
                                  env=_environment())
        if proc.returncode != 0:
            error = f"claude -p exited with code {proc.returncode}"
        else:
            text, served, tin, tout, cost = _parse(proc.stdout)
    except subprocess.TimeoutExpired:
        error = f"claude -p timed out after {timeout}s"
    except (json.JSONDecodeError, ValueError) as exc:
        error = f"unreadable claude output: {type(exc).__name__}"
    call = ClaudeCall(key=key, model=served, text=text if error is None else None,
                      input_tokens=tin, output_tokens=tout, cost_usd=cost,
                      latency_s=round(time.monotonic() - start, 3), error=error)
    _append(cache_dir, call)
    return call
