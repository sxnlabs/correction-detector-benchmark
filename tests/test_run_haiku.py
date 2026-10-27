from datetime import UTC, datetime

from cdbench.dataset import Message
from cdbench.haiku_runs import majority, measure_overhead, run_haiku


def msgs():
    return [Message(str(i), datetime(2026, 10, 1, tzinfo=UTC), f"texte {i}", "avant", i != 2)
            for i in range(3)]


def test_run_haiku_resumes(fake_claude, tmp_path):
    fake_claude.set(answer="correction")
    rows = run_haiku(msgs(), runs=3, variants=("message", "with_prior"), cache_dir=tmp_path,
                     workers=2)
    assert len(rows) == 2 * 3 * 2  # message 2 is not eligible
    first = fake_claude.calls
    run_haiku(msgs(), runs=3, variants=("message", "with_prior"), cache_dir=tmp_path, workers=2)
    assert fake_claude.calls == first


def test_errors_counted_not_labeled(fake_claude, tmp_path):
    fake_claude.set(fail=True)
    rows = run_haiku(msgs()[:1], runs=1, variants=("message",), cache_dir=tmp_path, workers=1)
    assert rows[0].label is None and rows[0].error is not None


def test_majority_ignores_failed_runs():
    assert majority([True, True, False]) is True
    assert majority([None, False, None]) is False
    assert majority([None, None]) is None


def test_measure_overhead(fake_claude, tmp_path):
    assert measure_overhead(tmp_path) == 110 - 10  # fake reports 110 input tokens


def test_off_format_answers_are_flagged(fake_claude, tmp_path):
    fake_claude.set(answer="Je ne peux pas répondre.")
    rows = run_haiku(msgs()[:1], runs=1, variants=("message",), cache_dir=tmp_path, workers=1)
    assert rows[0].off_format is True and rows[0].label is False
