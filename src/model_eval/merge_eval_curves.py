from __future__ import annotations

import argparse
import csv
from typing import Dict, List

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


def _project_rows(rows: List[Dict[str, str]], schema_columns: List[str]) -> List[Dict[str, str]]:
    projected: List[Dict[str, str]] = []
    for row in rows:
        out: Dict[str, str] = {}
        for col in schema_columns:
            value = row.get(col, "nan")
            out[col] = value if value != "" else "nan"
        projected.append(out)
    return projected


def main() -> None:
    args = parse_args()
    sft_rows = _read_csv(args.sft_structured_csv)
    if not sft_rows:
        raise RuntimeError("sft_structured_csv is empty.")
    schema_columns = list(sft_rows[0].keys())
    out_rows: List[Dict[str, str]] = []
    out_rows.extend(_project_rows(sft_rows, schema_columns))
    out_rows.extend(_project_rows(_read_csv(args.base_protocol_csv), schema_columns))
    out_rows.extend(_project_rows(_read_csv(args.base_task_think_csv), schema_columns))
    out_rows.extend(_project_rows(_read_csv(args.base_task_nothink_csv), schema_columns))

    out_rows.sort(key=lambda row: (int(str(row.get("model_step") or "0")), str(row.get("model_tag") or ""), str(row.get("model_path") or "")))

    write_csv(out_rows, args.out_csv)
    print(f"Saved merged comparison table to: {args.out_csv}")


if __name__ == "__main__":
    main()
