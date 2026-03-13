from __future__ import annotations

import argparse
from itertools import islice
from typing import Any, Dict, List, Tuple

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import GPTModel

from dataio import write_jsonl
from .common import extract_structured_output_text, load_dataset_split, load_generated_rows, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Input generated answers
    parser.add_argument("--generated_path", type=str, required=True, help="Generated answers JSONL or dataset path.")
    parser.add_argument("--split", type=str, default="train", help="Split name when generated_path is a DatasetDict.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum evaluated rows. -1 means all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input_mode",
        type=str,
        default="flat",
        choices=["flat", "structured"],
        help="Whether generated_path contains plain answers or structured generations.",
    )
    parser.add_argument(
        "--answer_source",
        type=str,
        default="answer",
        choices=["answer", "rationale_plus_answer"],
        help="Select the evaluated text source for flat mode.",
    )

    # DeepEval Hallucination metric
    parser.add_argument("--judge_model", type=str, default="gpt-5.2")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_concurrent", type=int, default=4)
    parser.add_argument("--throttle_value", type=float, default=3.0)

    # Outputs
    parser.add_argument("--metrics_out", type=str, required=True, help="Summary metrics CSV output path.")
    parser.add_argument("--details_out", type=str, default=None, help="Optional per-sample details JSONL path.")
    return parser.parse_args()


def _chunked(items: List[LLMTestCase], size: int) -> List[List[LLMTestCase]]:
    """Split test cases into fixed-size batches."""

    if size <= 0:
        return [items]
    iterator = iter(items)
    chunks: List[List[LLMTestCase]] = []
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            break
        chunks.append(batch)
    return chunks


def _build_metric(judge: GPTModel, threshold: float) -> HallucinationMetric:
    """Create a fresh hallucination metric instance for one DeepEval call."""

    return HallucinationMetric(
        threshold=threshold,
        model=judge,
        async_mode=True,
    )


def _detail_from_test_result(tr: Any) -> Dict[str, Any]:
    """Convert one DeepEval test result to a detail row."""

    meta = tr.additional_metadata or {}
    metric_data = tr.metrics_data[0] if tr.metrics_data else None
    return {
        "model_tag": meta.get("model_tag"),
        "model_step": meta.get("model_step"),
        "model_path": meta.get("model_path"),
        "eval_track": meta.get("eval_track"),
        "eval_variant": meta.get("eval_variant"),
        "sample_id": meta.get("sample_id"),
        "source_id": meta.get("source_id"),
        "structured_parse_ok": meta.get("structured_parse_ok"),
        "structured_answer_present": meta.get("structured_answer_present"),
        "structured_rationale_present": meta.get("structured_rationale_present"),
        "structured_extract_failed": meta.get("structured_extract_failed"),
        "input": tr.input,
        "actual_output": tr.actual_output,
        "context": tr.context,
        "metric_name": None if metric_data is None else metric_data.name,
        "score": None if metric_data is None else metric_data.score,
        "threshold": None if metric_data is None else metric_data.threshold,
        "success": None if metric_data is None else metric_data.success,
        "reason": None if metric_data is None else metric_data.reason,
        "evaluation_model": None if metric_data is None else metric_data.evaluation_model,
        "error": None if metric_data is None else metric_data.error,
        "evaluation_cost": None if metric_data is None else metric_data.evaluation_cost,
    }


def _error_detail_from_case(case: LLMTestCase, error: Exception) -> Dict[str, Any]:
    """Build a synthetic detail row when DeepEval raises before returning a test result."""

    meta = case.additional_metadata or {}
    return {
        "model_tag": meta.get("model_tag"),
        "model_step": meta.get("model_step"),
        "model_path": meta.get("model_path"),
        "eval_track": meta.get("eval_track"),
        "eval_variant": meta.get("eval_variant"),
        "sample_id": meta.get("sample_id"),
        "source_id": meta.get("source_id"),
        "structured_parse_ok": meta.get("structured_parse_ok"),
        "structured_answer_present": meta.get("structured_answer_present"),
        "structured_rationale_present": meta.get("structured_rationale_present"),
        "structured_extract_failed": meta.get("structured_extract_failed"),
        "input": case.input,
        "actual_output": case.actual_output,
        "context": case.context,
        "metric_name": "HallucinationMetric",
        "score": None,
        "threshold": None,
        "success": None,
        "reason": None,
        "evaluation_model": None,
        "error": f"{type(error).__name__}: {error}",
        "evaluation_cost": None,
    }


def _run_deepeval_once(
    *,
    cases: List[LLMTestCase],
    judge: GPTModel,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Run one DeepEval call and return parsed detail rows."""

    result = evaluate(
        test_cases=cases,
        metrics=[_build_metric(judge=judge, threshold=args.threshold)],
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
    return [_detail_from_test_result(tr) for tr in result.test_results]


def _run_deepeval_with_fallback(
    *,
    cases: List[LLMTestCase],
    judge: GPTModel,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Run DeepEval in batches and fall back to single-case evaluation on batch failure."""

    details: List[Dict[str, Any]] = []
    batch_size = max(1, int(args.max_concurrent))
    for batch in _chunked(cases, batch_size):
        try:
            details.extend(_run_deepeval_once(cases=batch, judge=judge, args=args))
            continue
        except Exception:
            pass
        for case in batch:
            try:
                details.extend(_run_deepeval_once(cases=[case], judge=judge, args=args))
            except Exception as error:  # pragma: no cover - exercised via integration-style mocking.
                details.append(_error_detail_from_case(case, error))
    return details


def run_deepeval_hallucination(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run DeepEval hallucination scoring and return summary and detail rows."""

    if args.input_mode == "structured":
        ds = load_dataset_split(data_path=args.generated_path, split=args.split).shuffle(seed=args.seed)
        rows: List[Dict[str, Any]] = []
        for idx, raw_row in enumerate(ds):
            row = dict(raw_row)
            answer_text, structured_meta = extract_structured_output_text(row)
            question = str(row.get("question") or "").strip()
            knowledge = str(row.get("knowledge") or "").strip()
            if not answer_text or not question or not knowledge:
                continue
            rows.append(
                {
                    "model_tag": str(row.get("model_tag") or row.get("tag") or "model"),
                    "model_step": int(row.get("model_step") or row.get("step") or 0),
                    "model_path": str(row.get("model_path") or row.get("model_name") or "model"),
                    "eval_track": str(row.get("eval_track") or ""),
                    "eval_variant": str(row.get("eval_variant") or ""),
                    "sample_id": int(row.get("sample_id") or idx),
                    "source_id": str(row.get("source_id") or row.get("id") or idx),
                    "question": question,
                    "knowledge": knowledge,
                    "answer": answer_text,
                    **structured_meta,
                }
            )
            if args.max_samples > 0 and len(rows) >= args.max_samples:
                break
    else:
        rows = load_generated_rows(
            generated_path=args.generated_path,
            split=args.split,
            max_samples=args.max_samples,
            seed=args.seed,
            answer_source=args.answer_source,
        )
    if not rows:
        raise RuntimeError("No valid generated rows found for DeepEval hallucination evaluation.")

    test_cases: List[LLMTestCase] = []
    for row in rows:
        test_cases.append(
            LLMTestCase(
                input=row["question"],
                actual_output=row["answer"],
                context=[row["knowledge"]],
                additional_metadata={
                    "model_tag": row["model_tag"],
                    "model_step": row["model_step"],
                    "model_path": row["model_path"],
                    "eval_track": row.get("eval_track"),
                    "eval_variant": row.get("eval_variant"),
                    "sample_id": row["sample_id"],
                    "source_id": row["source_id"],
                    "structured_parse_ok": row.get("structured_parse_ok"),
                    "structured_answer_present": row.get("structured_answer_present"),
                    "structured_rationale_present": row.get("structured_rationale_present"),
                    "structured_extract_failed": row.get("structured_extract_failed"),
                },
            )
        )

    judge = GPTModel(
        model=args.judge_model,
        temperature=0,
        generation_kwargs={
            # Increase output budget so structured JSON can finish
            "max_completion_tokens": 1024,
        },
    )
    
    details = _run_deepeval_with_fallback(cases=test_cases, judge=judge, args=args)

    grouped: Dict[Tuple[str, int, str, str, str], List[Dict[str, Any]]] = {}
    for row in details:
        key = (
            str(row["model_tag"]),
            int(row["model_step"]),
            str(row["model_path"]),
            str(row.get("eval_track") or ""),
            str(row.get("eval_variant") or ""),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for key, model_rows in sorted(grouped.items(), key=lambda kv: (kv[0][3], kv[0][1], kv[0][4], kv[0][0], kv[0][2])):
        scores = [float(x["score"]) for x in model_rows if x["score"] is not None]
        success = [bool(x["success"]) for x in model_rows if x["success"] is not None]
        errors = [x for x in model_rows if x["error"]]

        summary_rows.append(
            {
                "model_tag": key[0],
                "model_step": key[1],
                "model_path": key[2],
                "eval_track": key[3],
                "eval_variant": key[4],
                "num_cases": len(model_rows),
                "num_scored": len(scores),
                "deepeval_scored_rate": (len(scores) / len(model_rows)) if model_rows else float("nan"),
                "num_success_labeled": len(success),
                "deepeval_pass_count": sum(1 for x in success if x),
                "deepeval_pass_rate_on_scored": (sum(1 for x in success if x) / len(success)) if success else float("nan"),
                "num_errors": len(errors),
                "hallucination_score_mean": (sum(scores) / len(scores)) if scores else float("nan"),
                "judge_model": args.judge_model,
                "threshold": args.threshold,
                "max_concurrent": args.max_concurrent,
                "throttle_value": args.throttle_value,
            }
        )

    return summary_rows, details


def main() -> None:
    args = parse_args()
    summary_rows, details = run_deepeval_hallucination(args)

    write_csv(summary_rows, args.metrics_out)
    print(f"Saved metrics to: {args.metrics_out}")

    if args.details_out:
        write_jsonl(args.details_out, details)
        print(f"Saved details to: {args.details_out}")


if __name__ == "__main__":
    main()
