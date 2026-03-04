from __future__ import annotations

from typing import Any, Dict, Iterable, List

from dataio import write_json, write_jsonl


def partition_examples_by_data_split(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group prepared examples by their downstream data split."""

    partitions: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        split_name = str(row.get("data_split") or "").strip()
        if not split_name:
            continue
        partitions.setdefault(split_name, []).append(dict(row))
    return partitions


def write_partitioned_examples(rows: Iterable[Dict[str, Any]], out_dir: str, metrics_out: str | None = None) -> Dict[str, Any]:
    """Write one JSONL file per downstream data split."""

    partitions = partition_examples_by_data_split(rows)
    metrics = {
        "num_rows": sum(len(items) for items in partitions.values()),
        "splits": {name: len(items) for name, items in partitions.items()},
    }

    for split_name, items in partitions.items():
        write_jsonl(f"{out_dir}/{split_name}.jsonl", items)
    if metrics_out:
        write_json(metrics_out, metrics)
    return metrics
