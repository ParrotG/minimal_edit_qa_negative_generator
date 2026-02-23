from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

try:
    from src.dataio import read_jsonl, write_json
    from src.prompt import build_qa_answer_prefix
except ImportError:  # pragma: no cover - compatibility fallback for editable installs.
    from dataio import read_jsonl, write_json
    from prompt import build_qa_answer_prefix


@dataclass(frozen=True)
class LengthFilterConfig:
    """Length constraints for DPO samples."""

    max_prompt_tokens: int
    max_total_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_jsonl", type=str, required=True, help="Input ssqpg final pair JSONL path.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output path for main DPO DatasetDict.")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max_prompt_tokens", type=int, default=512)
    parser.add_argument("--max_total_tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=-1, help="<=0 means keeping all rows.")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--validation_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--overwrite_output", action="store_true")

    parser.add_argument(
        "--build_halueval_compare",
        action="store_true",
        help="Build a comparison dataset from original HaluEval rows indexed by the same ids.",
    )
    parser.add_argument(
        "--compare_output_dir",
        type=str,
        default=None,
        help="Output path for the optional HaluEval comparison DatasetDict.",
    )
    parser.add_argument("--halueval_dataset_name", type=str, default="pminervini/HaluEval")
    parser.add_argument("--halueval_subset", type=str, default="qa")
    parser.add_argument("--halueval_split", type=str, default="data")

    return parser.parse_args()


def _token_len(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def _normalize_ratio(train_ratio: float, validation_ratio: float, test_ratio: float) -> Tuple[float, float, float]:
    for name, value in (
        ("train_ratio", train_ratio),
        ("validation_ratio", validation_ratio),
        ("test_ratio", test_ratio),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")

    total = train_ratio + validation_ratio + test_ratio
    if total <= 0:
        raise ValueError("train_ratio + validation_ratio + test_ratio must be positive.")

    return train_ratio / total, validation_ratio / total, test_ratio / total


def _parse_source_id(row: Dict[str, Any], idx: int) -> str:
    return str(row.get("source_id") or row.get("id") or idx)


def _normalize_main_rows(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()
    metrics = {
        "num_input": 0,
        "drop_missing_fields": 0,
        "drop_duplicates": 0,
    }

    for idx, row in enumerate(rows):
        metrics["num_input"] += 1
        knowledge = str(row.get("knowledge") or "").strip()
        question = str(row.get("question") or "").strip()
        chosen = str(row.get("chosen") or "").strip()
        rejected = str(row.get("rejected") or "").strip()
        source_id = _parse_source_id(row, idx)

        if not knowledge or not question or not chosen or not rejected:
            metrics["drop_missing_fields"] += 1
            continue

        prompt = build_qa_answer_prefix(knowledge=knowledge, question=question)
        dedup_key = (prompt, chosen, rejected)
        if dedup_key in seen:
            metrics["drop_duplicates"] += 1
            continue
        seen.add(dedup_key)

        out.append(
            {
                "task": "qa",
                "source_id": source_id,
                "id": source_id,
                "knowledge": knowledge,
                "question": question,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    return out, metrics


def _filter_by_length(
    rows: Sequence[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    cfg: LengthFilterConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    kept: List[Dict[str, Any]] = []
    metrics = {
        "num_input": len(rows),
        "num_kept": 0,
        "drop_prompt_too_long": 0,
        "drop_total_too_long": 0,
    }

    for row in rows:
        prompt_len = _token_len(tokenizer, row["prompt"])
        chosen_len = _token_len(tokenizer, row["chosen"])
        rejected_len = _token_len(tokenizer, row["rejected"])

        if prompt_len > cfg.max_prompt_tokens:
            metrics["drop_prompt_too_long"] += 1
            continue
        if prompt_len + chosen_len > cfg.max_total_tokens or prompt_len + rejected_len > cfg.max_total_tokens:
            metrics["drop_total_too_long"] += 1
            continue

        kept.append(
            {
                **row,
                "prompt_tokens": int(prompt_len),
                "chosen_tokens": int(chosen_len),
                "rejected_tokens": int(rejected_len),
                "chosen_total_tokens": int(prompt_len + chosen_len),
                "rejected_total_tokens": int(prompt_len + rejected_len),
            }
        )

    metrics["num_kept"] = len(kept)
    return kept, metrics


def _split_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> DatasetDict:
    if not rows:
        raise ValueError("No rows available after preprocessing.")

    ds = Dataset.from_list(list(rows)).shuffle(seed=seed)
    n_rows = len(ds)
    if n_rows == 1:
        return DatasetDict({"train": ds})

    holdout_ratio = validation_ratio + test_ratio
    if holdout_ratio <= 0:
        return DatasetDict({"train": ds})

    holdout_size = max(1, int(round(n_rows * holdout_ratio)))
    holdout_size = min(holdout_size, n_rows - 1)
    split_main = ds.train_test_split(test_size=holdout_size, seed=seed)
    train_ds = split_main["train"]
    holdout_ds = split_main["test"]

    if test_ratio <= 0:
        return DatasetDict({"train": train_ds, "validation": holdout_ds})
    if validation_ratio <= 0:
        return DatasetDict({"train": train_ds, "test": holdout_ds})
    if len(holdout_ds) < 2:
        return DatasetDict({"train": train_ds, "validation": holdout_ds})

    test_fraction = test_ratio / (validation_ratio + test_ratio)
    holdout_split = holdout_ds.train_test_split(test_size=test_fraction, seed=seed)
    return DatasetDict(
        {
            "train": train_ds,
            "validation": holdout_split["train"],
            "test": holdout_split["test"],
        }
    )


def _prepare_output_dir(path: str, overwrite: bool) -> None:
    if os.path.isdir(path):
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Use --overwrite_output to replace it.")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _extract_halueval_ids(rows: Iterable[Dict[str, Any]]) -> Tuple[List[int], Dict[str, int]]:
    ids: List[int] = []
    seen: set[int] = set()
    metrics = {
        "num_input_rows": 0,
        "drop_invalid_id": 0,
        "drop_duplicate_id": 0,
    }
    for row in rows:
        metrics["num_input_rows"] += 1
        raw_id = str(row.get("id") or row.get("source_id") or "").strip()
        try:
            idx = int(raw_id)
        except ValueError:
            metrics["drop_invalid_id"] += 1
            continue
        if idx in seen:
            metrics["drop_duplicate_id"] += 1
            continue
        seen.add(idx)
        ids.append(idx)
    return ids, metrics


def _build_halueval_compare_rows(
    source_rows: Sequence[Dict[str, Any]],
    dataset_name: str,
    subset: str,
    split: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    ids, id_metrics = _extract_halueval_ids(source_rows)
    ds = load_dataset(dataset_name, subset, split=split)
    total = len(ds)

    out: List[Dict[str, Any]] = []
    metrics = {
        **id_metrics,
        "num_hf_rows": total,
        "num_selected_ids": len(ids),
        "drop_out_of_range": 0,
        "drop_missing_fields": 0,
        "num_kept": 0,
    }

    for idx in ids:
        if idx < 0 or idx >= total:
            metrics["drop_out_of_range"] += 1
            continue

        row = ds[idx]
        knowledge = str(row.get("knowledge") or "").strip()
        question = str(row.get("question") or "").strip()
        chosen = str(row.get("right_answer") or "").strip()
        rejected = str(row.get("hallucinated_answer") or "").strip()

        if not knowledge or not question or not chosen or not rejected:
            metrics["drop_missing_fields"] += 1
            continue

        out.append(
            {
                "task": "qa_halueval_compare",
                "source_id": str(idx),
                "id": str(idx),
                "knowledge": knowledge,
                "question": question,
                "prompt": build_qa_answer_prefix(knowledge=knowledge, question=question),
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    metrics["num_kept"] = len(out)
    return out, metrics


def _truncate_rows(rows: Sequence[Dict[str, Any]], max_samples: int) -> List[Dict[str, Any]]:
    if max_samples is None or max_samples <= 0:
        return list(rows)
    return list(rows[: max(0, max_samples)])


def _save_dataset_with_stats(
    *,
    output_dir: str,
    dataset: DatasetDict,
    stats: Dict[str, Any],
) -> None:
    dataset.save_to_disk(output_dir)
    write_json(os.path.join(output_dir, "prepare_stats.json"), stats)


def main() -> None:
    args = parse_args()
    train_ratio, validation_ratio, test_ratio = _normalize_ratio(
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    length_cfg = LengthFilterConfig(
        max_prompt_tokens=args.max_prompt_tokens,
        max_total_tokens=args.max_total_tokens,
    )

    raw_rows = list(read_jsonl(args.in_jsonl))
    normalized_rows, normalize_metrics = _normalize_main_rows(raw_rows)
    normalized_rows = _truncate_rows(normalized_rows, args.max_samples)
    filtered_rows, filter_metrics = _filter_by_length(normalized_rows, tokenizer, length_cfg)
    main_ds = _split_rows(
        filtered_rows,
        seed=args.seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    _prepare_output_dir(args.output_dir, overwrite=args.overwrite_output)
    main_stats = {
        "input_jsonl": args.in_jsonl,
        "tokenizer_name": args.tokenizer_name,
        "ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "normalize_metrics": normalize_metrics,
        "filter_metrics": filter_metrics,
        "split_sizes": {k: len(v) for k, v in main_ds.items()},
    }
    _save_dataset_with_stats(output_dir=args.output_dir, dataset=main_ds, stats=main_stats)
    print(f"Saved main DPO dataset to: {args.output_dir}")
    print(main_stats)

    if not args.build_halueval_compare:
        return

    compare_output_dir = args.compare_output_dir
    if not compare_output_dir:
        compare_output_dir = f"{args.output_dir.rstrip('/')}_halueval_compare"

    compare_rows, compare_row_metrics = _build_halueval_compare_rows(
        source_rows=normalized_rows,
        dataset_name=args.halueval_dataset_name,
        subset=args.halueval_subset,
        split=args.halueval_split,
    )
    compare_rows = _truncate_rows(compare_rows, args.max_samples)
    compare_filtered_rows, compare_filter_metrics = _filter_by_length(compare_rows, tokenizer, length_cfg)
    compare_ds = _split_rows(
        compare_filtered_rows,
        seed=args.seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    _prepare_output_dir(compare_output_dir, overwrite=args.overwrite_output)
    compare_stats = {
        "source_from_ids_of": args.in_jsonl,
        "halueval_dataset_name": args.halueval_dataset_name,
        "halueval_subset": args.halueval_subset,
        "halueval_split": args.halueval_split,
        "tokenizer_name": args.tokenizer_name,
        "ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "row_metrics": compare_row_metrics,
        "filter_metrics": compare_filter_metrics,
        "split_sizes": {k: len(v) for k, v in compare_ds.items()},
    }
    _save_dataset_with_stats(output_dir=compare_output_dir, dataset=compare_ds, stats=compare_stats)
    print(f"Saved HaluEval comparison dataset to: {compare_output_dir}")
    print(compare_stats)


if __name__ == "__main__":
    main()
