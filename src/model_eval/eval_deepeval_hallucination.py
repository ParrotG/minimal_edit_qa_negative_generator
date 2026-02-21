from __future__ import annotations

import argparse
from typing import Any, Dict, List, Tuple

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

from dataio import write_jsonl
from .common import load_generated_rows, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Input generated answers
    parser.add_argument("--generated_path", type=str, required=True, help="Generated answers JSONL or dataset path.")
    parser.add_argument("--split", type=str, default="train", help="Split name when generated_path is a DatasetDict.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum evaluated rows. -1 means all.")
    parser.add_argument("--seed", type=int, default=42)

    # DeepEval Hallucination metric
    parser.add_argument("--judge_model", type=str, default="gpt-4.1")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_concurrent", type=int, default=4)
    parser.add_argument("--throttle_value", type=float, default=5.0)

    # Outputs
    parser.add_argument("--metrics_out", type=str, required=True, help="Summary metrics CSV output path.")
    parser.add_argument("--details_out", type=str, default=None, help="Optional per-sample details JSONL path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = load_generated_rows(
        generated_path=args.generated_path,
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
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
    for key, model_rows in sorted(grouped.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        scores = [float(x["score"]) for x in model_rows if x["score"] is not None]
        success = [bool(x["success"]) for x in model_rows if x["success"] is not None]
        errors = [x for x in model_rows if x["error"]]

        summary_rows.append(
            {
                "model_tag": key[0],
                "model_step": key[1],
                "model_path": key[2],
                "num_cases": len(model_rows),
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

    write_csv(summary_rows, args.metrics_out)
    print(f"Saved metrics to: {args.metrics_out}")

    if args.details_out:
        write_jsonl(details, args.details_out)
        print(f"Saved details to: {args.details_out}")


if __name__ == "__main__":
    main()
