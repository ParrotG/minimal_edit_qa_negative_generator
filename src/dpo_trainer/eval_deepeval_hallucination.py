from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

try:
    from src.llm_textgen import GeneratorModelSpec, build_generator_model_specs, load_generator_from_spec
except ImportError:  # pragma: no cover - compatibility fallback for editable installs.
    from llm_textgen import GeneratorModelSpec, build_generator_model_specs, load_generator_from_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data_path", type=str, required=True, help="Dataset directory (save_to_disk) or JSON/JSONL file.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split name when data_path is a DatasetDict.")
    parser.add_argument("--task", type=str, default="qa", help="Task filter value. Empty to disable task filtering.")
    parser.add_argument("--max_samples", type=int, default=200, help="Maximum number of evaluated rows. -1 means all.")
    parser.add_argument("--seed", type=int, default=42)

    # LoRA selection: mutually exclusive
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="Single LoRA checkpoint path or merged model path.")
    parser.add_argument("--lora_ckpt_list_path", type=str, default=None, help="Directory containing checkpoint-* subdirectories.")
    parser.add_argument("--lora_is_merged", action="store_true", help="Set when --lora_ckpt_path points to a merged full model.")
    parser.add_argument(
        "--lora_list_are_merged",
        action="store_true",
        help="Set when checkpoints under --lora_ckpt_list_path are merged full models.",
    )
    parser.add_argument("--merge_lora", action="store_true", help="Merge adapter weights before generation.")

    # Generation
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--use_chat_template", action="store_true",default=True)

    # DeepEval Hallucination metric
    parser.add_argument("--judge_model", type=str, default="gpt-4.1")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_concurrent", type=int, default=4)
    parser.add_argument("--throttle_value", type=float, default=5.0)

    # Outputs
    parser.add_argument("--metrics_out", type=str, required=True, help="Summary metrics CSV output path.")
    parser.add_argument("--details_out", type=str, default=None, help="Optional per-sample details JSONL path.")
    return parser.parse_args()


def _normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _extract_eval_item(row: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    question = (row.get("question") or row.get("eval_question") or row.get("input") or "").strip()

    contexts = _normalize_list(row.get("eval_contexts"))
    if not contexts:
        contexts = _normalize_list(row.get("contexts"))
    if not contexts:
        contexts = _normalize_list(row.get("retrieval_context"))
    if not contexts:
        knowledge = (row.get("knowledge") or "").strip()
        if knowledge:
            contexts = [knowledge]

    if not question or not contexts:
        prompt = (row.get("prompt") or "").strip()
        if prompt and "\nQuestion:" in prompt:
            left, right = prompt.rsplit("\nQuestion:", 1)
            if not contexts and left.strip():
                contexts = [left.strip()]
            if not question:
                question = right.strip()

    if not question or not contexts:
        return None

    source_id = str(row.get("source_id") or row.get("id") or idx)
    knowledge = "\n\n".join(contexts)
    return {
        "sample_id": idx,
        "source_id": source_id,
        "question": question,
        "contexts": contexts,
        "knowledge": knowledge,
    }


def _load_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    path = args.data_path

    ds_obj: Any
    if os.path.isdir(path):
        ds_obj = load_from_disk(path)
        if isinstance(ds_obj, DatasetDict):
            if args.split not in ds_obj:
                raise ValueError(f"Split '{args.split}' not found in dataset dict. Available: {list(ds_obj.keys())}")
            ds = ds_obj[args.split]
        elif isinstance(ds_obj, Dataset):
            ds = ds_obj
        else:
            raise ValueError(f"Unsupported dataset object from {path}: {type(ds_obj)}")
    else:
        ds = load_dataset("json", data_files=path, split="train")

    if args.task.strip() and "task" in ds.column_names:
        task_value = args.task.strip()
        ds = ds.filter(lambda x: str(x.get("task", "")) == task_value)

    ds = ds.shuffle(seed=args.seed)
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        item = _extract_eval_item(row, idx)
        if item is not None:
            out.append(item)
    return out


def _write_csv(rows: Sequence[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    items = _load_rows(args)
    if not items:
        raise RuntimeError("No valid QA rows found after loading/filtering.")

    model_specs: List[GeneratorModelSpec] = build_generator_model_specs(
        base_model_name=args.base_model,
        lora_ckpt_path=args.lora_ckpt_path,
        lora_ckpt_list_path=args.lora_ckpt_list_path,
        lora_is_merged=bool(args.lora_is_merged),
        lora_list_are_merged=bool(args.lora_list_are_merged),
        include_base=True,
    )

    # Phase 1: generate outputs for all models first.
    generated_rows: List[Dict[str, Any]] = []
    for spec in model_specs:
        generator = load_generator_from_spec(
            spec=spec,
            base_model_name=args.base_model,
            merge_lora=args.merge_lora,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            use_chat_template=args.use_chat_template,
            seed=args.seed,
        )

        for start in range(0, len(items), args.batch_size):
            batch = items[start : start + args.batch_size]
            qa_items = [
                {
                    "knowledge": str(x["knowledge"]),
                    "question": str(x["question"]),
                }
                for x in batch
            ]

            answers = generator.generate_many_from_qa(
                qa_items=qa_items,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                use_chat_template=args.use_chat_template,
            )

            for sample, answer in zip(batch, answers):
                generated_rows.append(
                    {
                        "model_tag": spec.tag,
                        "model_step": int(spec.step),
                        "model_path": spec.display_name,
                        "sample_id": int(sample["sample_id"]),
                        "source_id": sample["source_id"],
                        "question": sample["question"],
                        "knowledge": sample["knowledge"],
                        "contexts": sample["contexts"],
                        "answer": answer,
                    }
                )

        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Phase 2: evaluate all generated rows together with DeepEval async controls.
    test_cases: List[LLMTestCase] = []
    for row in generated_rows:
        test_cases.append(
            LLMTestCase(
                input=row["question"],
                actual_output=row["answer"],
                context=row["contexts"],
                additional_metadata={
                    "model_tag": row["model_tag"],
                    "model_step": row["model_step"],
                    "model_path": row["model_path"],
                    "sample_id": row["sample_id"],
                    "source_id": row["source_id"],
                },
            )
        )

    metric = HallucinationMetric(
        threshold=args.threshold,
        model=args.judge_model,
        async_mode=True,
    )
    result = evaluate(
        test_cases=test_cases,
        metrics=[metric],
        async_config=AsyncConfig(
            run_async=True,
            throttle_value=float(args.throttle_value),
            max_concurrent=int(args.max_concurrent),
        ),
        display_config=DisplayConfig(
            show_indicator=True,
            print_results=False,
        ),
    )

    # Parse deepeval outputs into details and per-model summary.
    details: List[Dict[str, Any]] = []
    for tr in result.test_results:
        meta = tr.additional_metadata or {}
        m = tr.metrics_data[0] if tr.metrics_data else None
        details.append(
            {
                "model_tag": meta.get("model_tag"),
                "model_step": meta.get("model_step"),
                "model_path": meta.get("model_path"),
                "sample_id": meta.get("sample_id"),
                "source_id": meta.get("source_id"),
                "input": tr.input,
                "actual_output": tr.actual_output,
                "context": tr.context,
                "metric_name": None if m is None else m.name,
                "score": None if m is None else m.score,
                "threshold": None if m is None else m.threshold,
                "success": None if m is None else m.success,
                "reason": None if m is None else m.reason,
                "evaluation_model": None if m is None else m.evaluation_model,
                "error": None if m is None else m.error,
                "evaluation_cost": None if m is None else m.evaluation_cost,
            }
        )

    grouped: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}
    for row in details:
        key = (str(row["model_tag"]), int(row["model_step"]), str(row["model_path"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        scores = [float(x["score"]) for x in rows if x["score"] is not None]
        success = [bool(x["success"]) for x in rows if x["success"] is not None]
        errors = [x for x in rows if x["error"]]

        summary_rows.append(
            {
                "model_tag": key[0],
                "model_step": key[1],
                "model_path": key[2],
                "num_cases": len(rows),
                "num_scored": len(scores),
                "num_success_labeled": len(success),
                "num_errors": len(errors),
                "hallucination_score_mean": (sum(scores) / len(scores)) if scores else float("nan"),
                "pass_rate": (sum(1 for x in success if x) / len(success)) if success else float("nan"),
                "judge_model": args.judge_model,
                "threshold": args.threshold,
                "max_concurrent": args.max_concurrent,
                "throttle_value": args.throttle_value,
            }
        )

    _write_csv(summary_rows, args.metrics_out)
    print(f"Saved metrics to: {args.metrics_out}")

    if args.details_out:
        _write_jsonl(details, args.details_out)
        print(f"Saved details to: {args.details_out}")


if __name__ == "__main__":
    main()
