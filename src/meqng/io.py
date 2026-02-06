from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import orjson


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    """Read a JSONL file and yield records."""
    p = Path(path)
    with p.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield orjson.loads(line)


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    """Write records to a JSONL file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        for rec in records:
            f.write(orjson.dumps(rec))
            f.write(b"\n")


def write_json(path: str | Path, obj: Any) -> None:
    """Write an object as pretty JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
