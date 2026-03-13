from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from datasets import Dataset, DatasetDict

try:
    from src.qa_protocol import build_infer_prompt
    from src.qa_protocol.token_budget import count_text_tokens_batch
    from src.dataio import read_jsonl, write_json
    from src.project_config.resolve import resolve_training_args
except ImportError:  # pragma: no cover - compatibility fallback for editable installs.
    from qa_protocol import build_infer_prompt
    from qa_protocol.token_budget import count_text_tokens_batch
    from dataio import read_jsonl, write_json
    from project_config.resolve import resolve_training_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_jsonl",
        type=str,
        default=None,
        help="Optional single-input mode. If provided without per-split paths, rows are split by `data_split`.",
    )
    parser.add_argument(
        "--train_paths",
        nargs="*",
        default=[],
        help="One or more JSONL paths for train split (typically selected SFT records).",
    )
    parser.add_argument(
        "--validation_paths",
        nargs="*",
        default=[],
        help="One or more JSONL paths for validation split (typically selected SFT records).",
    )
    parser.add_argument(
        "--test_paths",
        nargs="*",
        default=[],
        help="One or more JSONL paths for test split (typically source partition outputs without completion).",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output DatasetDict.save_to_disk directory.")
    parser.add_argument("--overwrite_output", action="store_true", help="Overwrite the output directory if it exists.")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed for deterministic capped sampling.")
    parser.add_argument("--no_shuffle", action="store_true", help="Disable shuffle before capped sampling.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Maximum train rows after filtering.")
    parser.add_argument("--max_validation_samples", type=int, default=None, help="Maximum validation rows after filtering.")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Maximum test rows after filtering.")
    parser.add_argument(
        "--max_train_answerable_samples",
        type=int,
        default=None,
        help="Maximum answerable rows for train split.",
    )
    parser.add_argument(
        "--max_train_unanswerable_samples",
        type=int,
        default=None,
        help="Maximum unanswerable rows for train split.",
    )
    parser.add_argument(
        "--max_validation_answerable_samples",
        type=int,
        default=None,
        help="Maximum answerable rows for validation split.",
    )
    parser.add_argument(
        "--max_validation_unanswerable_samples",
        type=int,
        default=None,
        help="Maximum unanswerable rows for validation split.",
    )
    parser.add_argument(
        "--max_test_answerable_samples",
        type=int,
        default=None,
        help="Maximum answerable rows for test split.",
    )
    parser.add_argument(
        "--max_test_unanswerable_samples",
        type=int,
        default=None,
        help="Maximum unanswerable rows for test split.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Tokenizer used for token filtering.",
    )
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=None,
        help="Optional prompt token upper bound. <=0 disables this filter.",
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=None,
        help="Optional completion token upper bound. <=0 disables this filter.",
    )
    parser.add_argument("--metrics_out", type=str, default=None, help="Optional JSON metrics output path.")
    return resolve_training_args(parser.parse_args())


def _prepare_output_dir(path: str, overwrite: bool) -> None:
    if os.path.isdir(path):
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Use --overwrite_output to replace it.")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _rows_to_dataset(rows: List[Dict[str, Any]]) -> Dataset:
    return Dataset.from_list(rows) if rows else Dataset.from_list([])


def _stable_row_key(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    meta_answerability = ""
    if isinstance(metadata, Mapping):
        meta_answerability = str(metadata.get("answerability_label") or "").strip()
    compact = {
        "id": str(row.get("id") or ""),
        "source_id": str(row.get("source_id") or ""),
        "variant_id": str(row.get("variant_id") or ""),
        "answerability_label": str(row.get("answerability_label") or meta_answerability),
        "prompt": str(row.get("prompt") or ""),
        "completion": str(row.get("completion") or ""),
        "question": str(row.get("question") or ""),
        "knowledge": str(row.get("knowledge") or ""),
        "reference_answer": str(row.get("reference_answer") or ""),
    }
    return json.dumps(compact, ensure_ascii=False, sort_keys=True)


def _resolve_answerability_label(row: Mapping[str, Any]) -> str:
    label = str(row.get("answerability_label") or "").strip()
    if label:
        return label
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        return str(metadata.get("answerability_label") or "").strip()
    return ""


def _load_rows(paths: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    metrics = {"num_read": 0, "num_dedup_drop": 0}
    for path in paths:
        for record in read_jsonl(path):
            metrics["num_read"] += 1
            row = dict(record)
            key = _stable_row_key(row)
            if key in seen:
                metrics["num_dedup_drop"] += 1
                continue
            seen.add(key)
            rows.append(row)
    return rows, metrics


def _split_rows_from_legacy_input(rows: Iterable[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    split_rows: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        out = dict(row)
        split = str(out.get("data_split") or "train")
        if split == "train_sft_raw":
            split = "train"
        elif split not in {"train", "validation", "test"}:
            split = "train"
        split_rows.setdefault(split, []).append(out)
    return split_rows


def _normalize_row(row: MutableMapping[str, Any], split_name: str, require_completion: bool) -> Tuple[Dict[str, Any] | None, str | None]:
    out = dict(row)
    out["data_split"] = split_name

    answerability_label = _resolve_answerability_label(out)
    if answerability_label:
        out["answerability_label"] = answerability_label

    prompt = str(out.get("prompt") or "").strip()
    if not prompt:
        knowledge = str(out.get("knowledge") or "").strip()
        question = str(out.get("question") or "").strip()
        if knowledge and question:
            prompt = build_infer_prompt(knowledge=knowledge, question=question)
    if not prompt:
        return None, "missing_prompt"
    out["prompt"] = prompt

    completion = str(out.get("completion") or "").strip()
    if require_completion and not completion:
        return None, "missing_completion"
    out["completion"] = completion
    return out, None


def _normalize_split_rows(rows: Sequence[Mapping[str, Any]], split_name: str, require_completion: bool) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    kept: List[Dict[str, Any]] = []
    dropped = Counter()
    for row in rows:
        normalized, reason = _normalize_row(dict(row), split_name=split_name, require_completion=require_completion)
        if normalized is None:
            dropped[str(reason or "unknown")] += 1
            continue
        kept.append(normalized)
    metrics = {
        "num_input": len(rows),
        "num_kept": len(kept),
        "num_drop_missing_prompt": int(dropped.get("missing_prompt", 0)),
        "num_drop_missing_completion": int(dropped.get("missing_completion", 0)),
    }
    return kept, metrics


def _apply_token_filter(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer_name: str,
    max_prompt_tokens: int,
    max_completion_tokens: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if not rows:
        return [], {"num_input": 0, "num_kept": 0, "num_drop_prompt_over_budget": 0, "num_drop_completion_over_budget": 0}

    prompt_counts: List[int] = []
    completion_counts: List[int] = []
    if max_prompt_tokens > 0:
        prompt_counts = count_text_tokens_batch([str(row.get("prompt") or "") for row in rows], tokenizer_name=tokenizer_name)
    if max_completion_tokens > 0:
        completion_counts = count_text_tokens_batch(
            [str(row.get("completion") or "") for row in rows], tokenizer_name=tokenizer_name
        )

    kept: List[Dict[str, Any]] = []
    num_drop_prompt = 0
    num_drop_completion = 0
    for idx, row in enumerate(rows):
        prompt_tokens = prompt_counts[idx] if prompt_counts else None
        completion_tokens = completion_counts[idx] if completion_counts else None
        if prompt_tokens is not None and prompt_tokens > max_prompt_tokens:
            num_drop_prompt += 1
            continue
        if completion_tokens is not None and completion_tokens > max_completion_tokens:
            num_drop_completion += 1
            continue

        out = dict(row)
        if prompt_tokens is not None:
            out["prompt_tokens"] = int(prompt_tokens)
        if completion_tokens is not None:
            out["completion_tokens"] = int(completion_tokens)
        kept.append(out)

    metrics = {
        "num_input": len(rows),
        "num_kept": len(kept),
        "num_drop_prompt_over_budget": num_drop_prompt,
        "num_drop_completion_over_budget": num_drop_completion,
    }
    return kept, metrics


def _sample_split_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split_name: str,
    seed: int,
    shuffle: bool,
    max_total: int,
    max_answerable: int,
    max_unanswerable: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    working = [dict(row) for row in rows]
    split_seed_offset = {"train": 11, "validation": 23, "test": 37}.get(split_name, 0)
    if shuffle:
        random.Random(seed + split_seed_offset).shuffle(working)

    kept: List[Dict[str, Any]] = []
    count_answerable = 0
    count_unanswerable = 0
    drop_label_cap = 0
    for row in working:
        label = _resolve_answerability_label(row)
        if label == "answerable" and max_answerable > -1 and count_answerable >= max_answerable:
            drop_label_cap += 1
            continue
        if label == "unanswerable" and max_unanswerable > -1 and count_unanswerable >= max_unanswerable:
            drop_label_cap += 1
            continue
        kept.append(row)
        if label == "answerable":
            count_answerable += 1
        elif label == "unanswerable":
            count_unanswerable += 1

    drop_total_cap = 0
    if max_total > -1 and len(kept) > max_total:
        drop_total_cap = len(kept) - max_total
        kept = kept[:max_total]

    label_counter = Counter(_resolve_answerability_label(row) or "unknown" for row in kept)
    metrics = {
        "num_input": len(rows),
        "num_kept": len(kept),
        "num_drop_by_label_cap": drop_label_cap,
        "num_drop_by_total_cap": drop_total_cap,
        "num_answerable_kept": int(label_counter.get("answerable", 0)),
        "num_unanswerable_kept": int(label_counter.get("unanswerable", 0)),
        "num_unknown_label_kept": int(label_counter.get("unknown", 0)),
    }
    return kept, metrics


def main() -> None:
    args = parse_args()

    use_explicit_paths = bool(args.train_paths or args.validation_paths or args.test_paths)
    if not use_explicit_paths and not args.in_jsonl:
        raise ValueError("Provide either --in_jsonl or any of --train_paths/--validation_paths/--test_paths.")

    io_metrics: Dict[str, Any] = {}
    if use_explicit_paths:
        train_raw, train_io_metrics = _load_rows(args.train_paths)
        validation_raw, validation_io_metrics = _load_rows(args.validation_paths)
        test_raw, test_io_metrics = _load_rows(args.test_paths)
        split_rows_raw: Dict[str, List[Dict[str, Any]]] = {
            "train": train_raw,
            "validation": validation_raw,
            "test": test_raw,
        }
        io_metrics = {
            "train": train_io_metrics,
            "validation": validation_io_metrics,
            "test": test_io_metrics,
        }
    else:
        legacy_rows = list(read_jsonl(args.in_jsonl))
        if not legacy_rows:
            raise RuntimeError("Input JSONL has no rows.")
        split_rows_raw = _split_rows_from_legacy_input(legacy_rows)
        io_metrics = {"legacy_num_rows": len(legacy_rows)}

    normalized_train, train_norm_metrics = _normalize_split_rows(
        split_rows_raw.get("train", []), split_name="train", require_completion=True
    )
    normalized_validation, validation_norm_metrics = _normalize_split_rows(
        split_rows_raw.get("validation", []), split_name="validation", require_completion=True
    )
    normalized_test, test_norm_metrics = _normalize_split_rows(
        split_rows_raw.get("test", []), split_name="test", require_completion=False
    )

    filtered_train, train_token_metrics = _apply_token_filter(
        normalized_train,
        tokenizer_name=args.tokenizer_name,
        max_prompt_tokens=args.max_prompt_tokens,
        max_completion_tokens=args.max_completion_tokens,
    )
    filtered_validation, validation_token_metrics = _apply_token_filter(
        normalized_validation,
        tokenizer_name=args.tokenizer_name,
        max_prompt_tokens=args.max_prompt_tokens,
        max_completion_tokens=args.max_completion_tokens,
    )
    filtered_test, test_token_metrics = _apply_token_filter(
        normalized_test,
        tokenizer_name=args.tokenizer_name,
        max_prompt_tokens=args.max_prompt_tokens,
        max_completion_tokens=args.max_completion_tokens,
    )

    sampled_train, train_sample_metrics = _sample_split_rows(
        filtered_train,
        split_name="train",
        seed=args.seed,
        shuffle=not args.no_shuffle,
        max_total=args.max_train_samples,
        max_answerable=args.max_train_answerable_samples,
        max_unanswerable=args.max_train_unanswerable_samples,
    )
    sampled_validation, validation_sample_metrics = _sample_split_rows(
        filtered_validation,
        split_name="validation",
        seed=args.seed,
        shuffle=not args.no_shuffle,
        max_total=args.max_validation_samples,
        max_answerable=args.max_validation_answerable_samples,
        max_unanswerable=args.max_validation_unanswerable_samples,
    )
    sampled_test, test_sample_metrics = _sample_split_rows(
        filtered_test,
        split_name="test",
        seed=args.seed,
        shuffle=not args.no_shuffle,
        max_total=args.max_test_samples,
        max_answerable=args.max_test_answerable_samples,
        max_unanswerable=args.max_test_unanswerable_samples,
    )

    if not sampled_train:
        raise RuntimeError("Train split is empty after normalization/filtering/sampling.")
    if not sampled_validation:
        raise RuntimeError("Validation split is empty after normalization/filtering/sampling.")

    split_rows: Dict[str, List[Dict[str, Any]]] = {
        "train": sampled_train,
        "validation": sampled_validation,
    }
    if sampled_test:
        split_rows["test"] = sampled_test

    ds_dict = DatasetDict({split: _rows_to_dataset(items) for split, items in split_rows.items()})
    _prepare_output_dir(args.output_dir, overwrite=args.overwrite_output)
    ds_dict.save_to_disk(args.output_dir)

    metrics = {
        "inputs": io_metrics,
        "normalization": {
            "train": train_norm_metrics,
            "validation": validation_norm_metrics,
            "test": test_norm_metrics,
        },
        "token_filter": {
            "tokenizer_name": args.tokenizer_name,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_completion_tokens": args.max_completion_tokens,
            "train": train_token_metrics,
            "validation": validation_token_metrics,
            "test": test_token_metrics,
        },
        "sampling": {
            "seed": args.seed,
            "shuffle": not args.no_shuffle,
            "train": train_sample_metrics,
            "validation": validation_sample_metrics,
            "test": test_sample_metrics,
        },
        "splits": {name: len(items) for name, items in split_rows.items()},
        "output_dir": args.output_dir,
    }
    if args.metrics_out:
        write_json(args.metrics_out, metrics)
    print(f"Saved SFT dataset to: {args.output_dir}")


if __name__ == "__main__":
    main()
