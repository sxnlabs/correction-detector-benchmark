from datetime import UTC, datetime

import pytest

from cdbench.dataset import Message
from cdbench.privacy import PrivacyLeak, assert_no_message_text

SECRET = "le client Martin attend toujours la facture de septembre depuis lundi"


def m(text):
    return Message("1", datetime(2026, 10, 1, tzinfo=UTC), text, None, True)


def test_privacy_guard_catches_window(tmp_path):
    leak = tmp_path / "summary.json"
    leak.write_text('{"note": "...' + SECRET[10:45] + '..."}')
    with pytest.raises(PrivacyLeak) as exc:
        assert_no_message_text([tmp_path], [m(SECRET)])
    assert SECRET[10:45] not in str(exc.value)
    assert "summary.json" in str(exc.value)


def test_privacy_guard_passes_clean_files(tmp_path):
    (tmp_path / "table.md").write_text("| regex | 0.25 | 0.90 |")
    assert_no_message_text([tmp_path], [m(SECRET)])


def test_short_messages_only_match_whole(tmp_path):
    (tmp_path / "t.md").write_text("ok merci, rapport genere")
    assert_no_message_text([tmp_path], [m("ok"), m("merci")])  # too short to identify
    (tmp_path / "u.md").write_text("on lit : envoie le devis Dupont ce soir")
    with pytest.raises(PrivacyLeak):
        assert_no_message_text([tmp_path], [m("envoie le devis Dupont")])


def test_binary_png_is_scanned_as_text_safely(tmp_path):
    (tmp_path / "plot.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00binary")
    assert_no_message_text([tmp_path], [m(SECRET)])
