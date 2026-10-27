"""JSONL files of dataclasses in the private data dir."""

import json
from dataclasses import asdict
from pathlib import Path


def write_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def read_rows(path: Path, cls) -> list:
    with path.open(encoding="utf-8") as f:
        return [cls(**json.loads(line)) for line in f if line.strip()]
