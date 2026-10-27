from cdbench.claude import run_claude


def test_run_claude_parses_json(fake_claude, tmp_path):
    fake_claude.set(answer="correction")
    call = run_claude("prompt", model="haiku", key="k1", cache_dir=tmp_path / "cache")
    assert call.text == "correction"
    assert call.model == "claude-haiku-5-5"
    assert call.input_tokens == 110
    assert call.output_tokens == 5
    assert call.cost_usd == 0.001
    assert call.error is None
    assert call.latency_s >= 0


def test_run_claude_uses_cache(fake_claude, tmp_path):
    cache = tmp_path / "cache"
    run_claude("prompt", model="haiku", key="k1", cache_dir=cache)
    run_claude("prompt", model="haiku", key="k1", cache_dir=cache)
    assert fake_claude.calls == 1


def test_failed_call_is_not_a_label(fake_claude, tmp_path):
    cache = tmp_path / "cache"
    fake_claude.set(fail=True)
    call = run_claude("prompt", model="haiku", key="k1", cache_dir=cache)
    assert call.text is None
    assert call.error is not None
    assert "prompt" not in call.error  # never echo the prompt (private text)
    fake_claude.set()
    retry = run_claude("prompt", model="haiku", key="k1", cache_dir=cache)
    assert retry.error is None  # a cached error is retried, not reused
    assert fake_claude.calls == 2


def test_corrupt_cache_line_is_skipped(fake_claude, tmp_path):
    cache = tmp_path / "cache"
    run_claude("prompt", model="haiku", key="k1", cache_dir=cache)
    with (cache / "calls.jsonl").open("a") as f:
        f.write('{"key": "k2", "trunc')  # a run killed mid-write
    again = run_claude("prompt", model="haiku", key="k1", cache_dir=cache)
    assert again.error is None and fake_claude.calls == 1
