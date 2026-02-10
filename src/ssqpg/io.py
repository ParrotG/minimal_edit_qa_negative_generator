from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

import orjson


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    """Read JSONL records from disk."""

    p = Path(path)
    with p.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield orjson.loads(line)


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    """Write JSONL records to disk."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        for rec in records:
            f.write(orjson.dumps(rec))
            f.write(b"\n")


def write_json(path: str | Path, payload: Any) -> None:
    """Write JSON payload with pretty formatting."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
