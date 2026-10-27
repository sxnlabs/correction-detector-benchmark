"""Scan the whole repo against the real messages, when the private data is present."""

import os
from pathlib import Path

import pytest

from cdbench.dataset import load_messages
from cdbench.privacy import assert_no_message_text

DATA = Path(os.environ.get("CDBENCH_DATA_DIR",
                           Path.home() / "Hermes/state/correction-benchmark")).expanduser()
REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (DATA / "snapshot.jsonl").exists(), reason="private data not here")
def test_repo_contains_no_message_text():
    paths = [REPO / p for p in ("README.md", "PROVENANCE.md", "src", "tests", "scripts",
                                "results")]
    assert_no_message_text([p for p in paths if p.exists()], load_messages(DATA))
