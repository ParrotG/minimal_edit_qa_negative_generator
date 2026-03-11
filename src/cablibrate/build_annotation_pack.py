from __future__ import annotations

import argparse
from datetime import datetime
from typing import List

from dataio import write_json, write_jsonl
from project_config import PROJECT_SETTINGS

from .common import build_annotation_pack_rows, load_dataset_split, summarize_annotation_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Input JSONL or DatasetDict path.")
    parser.add_argument("--split", type=str, default="train", help="Split name when data_path is a DatasetDict.")
    parser.add_argument(
        "--task_types",
        type=str,
        default=",".join(PROJECT_SETTINGS.calibration.annotation_task_types),
        help="Comma-separated calibration task types.",
    )
    parser.add_argument("--max_samples_per_task", type=int, default=200, help="Maximum sampled rows for each task type.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pack_id", type=str, default="", help="Optional explicit pack identifier.")
    parser.add_argument("--out_jsonl", type=str, required=True, help="Annotation-pack JSONL path.")
    parser.add_argument("--metrics_out", type=str, default="", help="Optional metrics JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_types: List[str] = [token.strip() for token in str(args.task_types or "").split(",") if token.strip()]
    if not task_types:
        raise RuntimeError("No task types provided for annotation-pack building.")

    ds = load_dataset_split(data_path=args.data_path, split=args.split)
    rows = [dict(row) for row in ds]
    pack_id = str(args.pack_id or datetime.now().strftime("calibration_pack_%Y%m%d_%H%M%S"))
    pack_rows, metrics = build_annotation_pack_rows(
        rows=rows,
        task_types=task_types,
        max_samples_per_task=int(args.max_samples_per_task),
        seed=int(args.seed),
        pack_id=pack_id,
    )
    if not pack_rows:
        raise RuntimeError("No annotation-pack rows could be constructed from the input data.")

    write_jsonl(args.out_jsonl, pack_rows)
    if str(args.metrics_out or "").strip():
        write_json(
            args.metrics_out,
            {
                **metrics,
                "summary": summarize_annotation_labels(pack_rows),
            },
        )
    print(f"Saved annotation pack: {args.out_jsonl}")


if __name__ == "__main__":
    main()
