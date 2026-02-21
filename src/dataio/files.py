from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import orjson


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    """Read JSONL records from disk as a streaming iterator."""

    p = Path(path)
    with p.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield orjson.loads(line)


def read_jsonl_list(path: str | Path) -> List[Dict[str, Any]]:
    """Read all JSONL records into a list."""

    return list(read_jsonl(path))


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    """Write records to JSONL with stable UTF-8 encoding."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        for record in records:
            f.write(orjson.dumps(record))
            f.write(b"\n")


def read_json(path: str | Path) -> Any:
    """Read a JSON file into a Python object."""

    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    """Write a Python object as pretty JSON."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
