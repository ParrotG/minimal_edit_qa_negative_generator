from __future__ import annotations

import argparse
import os
import shutil
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict

try:
    from src.dataio import read_jsonl, write_json
except ImportError:  # pragma: no cover - compatibility fallback for editable installs.
    from dataio import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_jsonl", type=str, required=True, help="Input SFT records JSONL path.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output DatasetDict.save_to_disk directory.")
    parser.add_argument("--overwrite_output", action="store_true", help="Overwrite the output directory if it exists.")
    parser.add_argument("--metrics_out", type=str, default=None, help="Optional JSON metrics output path.")
    return parser.parse_args()


def _prepare_output_dir(path: str, overwrite: bool) -> None:
    if os.path.isdir(path):
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Use --overwrite_output to replace it.")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _rows_to_dataset(rows: List[Dict[str, Any]]) -> Dataset:
    return Dataset.from_list(rows) if rows else Dataset.from_list([])


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(args.in_jsonl))
    if not rows:
        raise RuntimeError("Input JSONL has no rows.")

    split_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        split = str(row.get("split") or "train")
        if split == "train_sft_raw":
            split = "train"
        split_rows.setdefault(split, []).append(dict(row))

    ds_dict = DatasetDict({split: _rows_to_dataset(items) for split, items in split_rows.items()})
    _prepare_output_dir(args.output_dir, overwrite=args.overwrite_output)
    ds_dict.save_to_disk(args.output_dir)

    metrics = {
        "num_rows": len(rows),
        "splits": {name: len(items) for name, items in split_rows.items()},
        "output_dir": args.output_dir,
    }
    if args.metrics_out:
        write_json(args.metrics_out, metrics)
    print(f"Saved SFT dataset to: {args.output_dir}")


if __name__ == "__main__":
    main()
