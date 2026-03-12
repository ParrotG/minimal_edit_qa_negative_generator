from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dataio import write_json, write_jsonl
from project_config import PROJECT_SETTINGS

from .common import build_annotation_pack_rows, load_dataset_split, summarize_annotation_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        action="append",
        required=True,
        help="Input JSONL or DatasetDict path. Repeat this flag to sample from multiple sources.",
    )
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


def _row_identity(row: Dict[str, Any]) -> tuple[str, str, int, str, str, str]:
    """Build a stable row identity for detail/generation sidecar joins."""

    return (
        str(row.get("source_id") or ""),
        str(row.get("model_tag") or ""),
        int(row.get("model_step") or 0),
        str(row.get("model_path") or ""),
        str(row.get("eval_track") or ""),
        str(row.get("eval_variant") or ""),
    )


def _enrich_rows_from_sibling_generations(data_path: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Backfill missing detail fields from a sibling generations file when present."""

    source_path = Path(data_path)
    if not source_path.name.endswith("_details.jsonl"):
        return rows
    sibling_path = source_path.with_name(source_path.name.replace("_details.jsonl", "_generations.jsonl"))
    if not sibling_path.exists():
        return rows

    sibling_rows = [dict(row) for row in load_dataset_split(data_path=str(sibling_path), split="train")]
    sibling_by_key = {_row_identity(row): row for row in sibling_rows}
    enriched_rows: List[Dict[str, Any]] = []
    for row in rows:
        sibling = sibling_by_key.get(_row_identity(row))
        if sibling is None:
            enriched_rows.append(row)
            continue
        enriched_rows.append(
            {
                **sibling,
                **row,
                "question": row.get("question") or sibling.get("question"),
                "knowledge": row.get("knowledge") or sibling.get("knowledge"),
                "reference_answer": row.get("reference_answer") or sibling.get("reference_answer"),
            }
        )
    return enriched_rows


def main() -> None:
    args = parse_args()
    task_types: List[str] = [token.strip() for token in str(args.task_types or "").split(",") if token.strip()]
    if not task_types:
        raise RuntimeError("No task types provided for annotation-pack building.")

    rows: List[Dict[str, Any]] = []
    input_metrics: List[Dict[str, Any]] = []
    for data_path in args.data_path:
        ds = load_dataset_split(data_path=data_path, split=args.split)
        source_rows = [dict(row) for row in ds]
        source_rows = _enrich_rows_from_sibling_generations(data_path, source_rows)
        source_rows = [{**dict(row), "input_source_path": str(data_path)} for row in source_rows]
        rows.extend(source_rows)
        input_metrics.append(
            {
                "input_source_path": str(data_path),
                "num_input_rows": len(source_rows),
            }
        )
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
                "input_sources": input_metrics,
                "summary": summarize_annotation_labels(pack_rows),
            },
        )
    print(f"Saved annotation pack: {args.out_jsonl}")


if __name__ == "__main__":
    main()
