from __future__ import annotations

import argparse
import csv
from typing import Dict, List, Tuple

from .common import write_csv


KEY_FIELDS = ("model_tag", "model_step", "model_path")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_structured_csv", type=str, required=True)
    parser.add_argument("--base_protocol_csv", type=str, required=True)
    parser.add_argument("--base_task_think_csv", type=str, required=True)
    parser.add_argument("--base_task_nothink_csv", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    return parser.parse_args()


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _merge_rows(
    merged: Dict[Tuple[str, str, str], Dict[str, str]],
    *,
    rows: List[Dict[str, str]],
    prefix: str,
) -> None:
    for row in rows:
        key = tuple(str(row.get(field) or "") for field in KEY_FIELDS)
        target = merged.setdefault(
            key,
            {
                "model_tag": key[0],
                "model_step": key[1],
                "model_path": key[2],
            },
        )
        for field, value in row.items():
            if field in KEY_FIELDS:
                continue
            target[f"{prefix}__{field}"] = value if value != "" else "nan"


def main() -> None:
    args = parse_args()
    merged: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    _merge_rows(merged, rows=_read_csv(args.sft_structured_csv), prefix="sft_structured")
    _merge_rows(merged, rows=_read_csv(args.base_protocol_csv), prefix="base_protocol")
    _merge_rows(merged, rows=_read_csv(args.base_task_think_csv), prefix="base_task_think")
    _merge_rows(merged, rows=_read_csv(args.base_task_nothink_csv), prefix="base_task_nothink")

    all_columns = set()
    for row in merged.values():
        all_columns.update(row.keys())
    ordered_metric_columns = sorted(col for col in all_columns if col not in KEY_FIELDS)

    out_rows: List[Dict[str, str]] = []
    for key in sorted(merged.keys(), key=lambda item: (int(item[1] or 0), item[0], item[2])):
        row = dict(merged[key])
        for col in ordered_metric_columns:
            row.setdefault(col, "nan")
        out_rows.append(row)

    write_csv(out_rows, args.out_csv)
    print(f"Saved merged comparison table to: {args.out_csv}")


if __name__ == "__main__":
    main()
