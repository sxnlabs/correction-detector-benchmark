import pytest

from cdbench.dataset import load_messages, snapshot
from tests.conftest import line


def _load(write_log, tmp_path, entries):
    data = tmp_path / "data"
    snapshot(write_log(entries), data)
    return load_messages(data)


def test_slash_commands_are_out_of_scope(write_log, tmp_path):
    msgs = _load(write_log, tmp_path, [line(0, "/focus projet", update_id=1),
                                       line(1, "fais le rapport", update_id=2)])
    assert [m.id for m in msgs] == ["2"]


def test_prior_reply_window(write_log, tmp_path):
    long_reply = "x" * 250 + "y" * 300
    msgs = _load(write_log, tmp_path, [
        line(0, "premier message", reply=long_reply, update_id=1),
        line(29, "second message", reply="r2", update_id=2),
        line(60, "troisieme message", update_id=3),
    ])
    by_id = {m.id: m for m in msgs}
    assert by_id["1"].prior_reply is None
    # production keeps the first 500 characters of the reply, the prompt the last 300 of those
    assert by_id["2"].prior_reply == "x" * 50 + "y" * 250
    assert by_id["3"].prior_reply is None  # previous reply 31 minutes earlier


def test_text_truncated_at_2000(write_log, tmp_path):
    msgs = _load(write_log, tmp_path, [line(0, "a" * 2500, update_id=1)])
    assert len(msgs[0].text) == 2000


def test_llm_eligibility(write_log, tmp_path):
    msgs = _load(write_log, tmp_path, [
        line(0, "ok", update_id=1),
        line(1, "Non", update_id=2),
        line(2, "\U0001F44D\U0001F44D\U0001F44D", update_id=3),
        line(3, "tu as oublie le fichier", update_id=4),
    ])
    by_id = {m.id: m for m in msgs}
    assert by_id["1"].llm_eligible is False  # shorter than LLM_MIN_CHARS
    assert by_id["2"].llm_eligible is False  # bare "Non" is ignored
    assert by_id["3"].llm_eligible is False  # 3 characters
    assert by_id["4"].llm_eligible is True


def test_id_falls_back_to_timestamp(write_log, tmp_path):
    msgs = _load(write_log, tmp_path, [line(0, "sans identifiant")])
    assert msgs[0].id == "2026-10-01T09:00:00+00:00"


def test_snapshot_refuses_overwrite(write_log, tmp_path):
    data = tmp_path / "data"
    log = write_log([line(0, "message", update_id=1)])
    snapshot(log, data)
    with pytest.raises(FileExistsError):
        snapshot(log, data)
