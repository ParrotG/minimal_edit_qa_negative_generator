from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List, Optional, Sequence

from dataio import write_jsonl

try:
    from src.nli_judge.config import NLIConfig
    from src.nli_judge.nli import NLIVerifier
    from src.prompt import build_qa_premise
except ImportError:  # pragma: no cover - compatibility fallback for editable installs.
    from nli_judge.config import NLIConfig
    from nli_judge.nli import NLIVerifier
    from prompt import build_qa_premise

from .common import load_dataset_split, normalize_list, safe_int, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--deepeval_details_path",
        type=str,
        required=True,
        help="DeepEval details path (JSONL or save_to_disk dataset).",
    )
    parser.add_argument("--split", type=str, default="train", help="Split name when details path is a DatasetDict.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum rows to score. -1 means all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_error_rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip rows where DeepEval reported non-empty error.",
    )
    parser.add_argument(
        "--deepeval_success_means_supported",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True, DeepEval success=True is treated as faithful/supported.",
    )

    parser.add_argument("--nli_model_name", type=str, default=NLIConfig.model_name)
    parser.add_argument("--nli_device", type=str, default=NLIConfig.device)
    parser.add_argument("--nli_batch_size", type=int, default=NLIConfig.batch_size)
    parser.add_argument("--nli_max_length", type=int, default=NLIConfig.max_length)
    parser.add_argument(
        "--nli_fp16",
        action=argparse.BooleanOptionalAction,
        default=NLIConfig.fp16,
        help="Whether to enable fp16 for NLI scoring on CUDA.",
    )

    parser.add_argument("--out_jsonl", type=str, required=True, help="Per-sample scored output JSONL path.")
    parser.add_argument("--out_csv", type=str, default="", help="Optional summary CSV path.")
    return parser.parse_args()


def _safe_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _clipped_logit(p: float, eps: float = 1e-12) -> float:
    q = min(max(float(p), eps), 1.0 - eps)
    return math.log(q / (1.0 - q))


def _load_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    ds = load_dataset_split(data_path=args.deepeval_details_path, split=args.split).shuffle(seed=args.seed)

    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        error_text = str(row.get("error") or "").strip()
        if args.skip_error_rows and error_text:
            continue

        question = str(row.get("input") or "").strip()
        answer = str(row.get("actual_output") or "").strip()
        contexts = normalize_list(row.get("context"))
        knowledge = "\n\n".join(contexts).strip()
        if not question or not answer or not knowledge:
            continue

        deepeval_success = _safe_bool(row.get("success"))
        if deepeval_success is not None and not args.deepeval_success_means_supported:
            deepeval_success = not deepeval_success

        out.append(
            {
                "model_tag": str(row.get("model_tag") or "model"),
                "model_step": safe_int(row.get("model_step"), 0),
                "model_path": str(row.get("model_path") or row.get("model_tag") or "model"),
                "sample_id": safe_int(row.get("sample_id"), idx),
                "source_id": str(row.get("source_id") or safe_int(row.get("sample_id"), idx)),
                "question": question,
                "knowledge": knowledge,
                "answer": answer,
                "deepeval_score": _safe_float(row.get("score")),
                "deepeval_threshold": _safe_float(row.get("threshold")),
                "deepeval_success": deepeval_success,
                "deepeval_error": error_text,
                "deepeval_metric_name": str(row.get("metric_name") or ""),
                "deepeval_reason": str(row.get("reason") or ""),
                "deepeval_evaluation_model": str(row.get("evaluation_model") or ""),
            }
        )
        if args.max_samples > 0 and len(out) >= args.max_samples:
            break
    return out


def _label_by_argmax(entail: float, neutral: float, contradict: float) -> str:
    if entail >= neutral and entail >= contradict:
        return "entail"
    if neutral >= entail and neutral >= contradict:
        return "neutral"
    return "contradict"


def main() -> None:
    args = parse_args()
    rows = _load_rows(args)
    if not rows:
        raise RuntimeError("No valid rows found from DeepEval details input.")

    verifier = NLIVerifier(
        model_name=args.nli_model_name,
        device=args.nli_device,
        batch_size=args.nli_batch_size,
        max_length=args.nli_max_length,
        fp16=args.nli_fp16,
    )

    premises = [build_qa_premise(knowledge=r["knowledge"], question=r["question"]) for r in rows]
    hypotheses = [r["answer"] for r in rows]
    scores = verifier.score(premises, hypotheses)

    scored_rows: List[Dict[str, Any]] = []
    for row, score in zip(rows, scores):
        entail = float(score.entail)
        neutral = float(score.neutral)
        contradict = float(score.contradict)
        label = _label_by_argmax(entail=entail, neutral=neutral, contradict=contradict)

        logit_entail = _clipped_logit(entail)
        logit_neutral = _clipped_logit(neutral)
        logit_contradict = _clipped_logit(contradict)
        margin_logit = logit_entail - max(logit_neutral, logit_contradict)

        scored_rows.append(
            {
                **row,
                "nli_entail": entail,
                "nli_neutral": neutral,
                "nli_contradict": contradict,
                "nli_argmax_label": label,
                "nli_argmax_prob": max(entail, neutral, contradict),
                "nli_logit_entail": logit_entail,
                "nli_logit_neutral": logit_neutral,
                "nli_logit_contradict": logit_contradict,
                "nli_margin_logit_entail_vs_max_other": margin_logit,
            }
        )

    write_jsonl(args.out_jsonl, scored_rows)
    print(f"Saved NLI-scored rows: {args.out_jsonl}")

    if args.out_csv.strip():
        summary = {
            "num_rows": len(scored_rows),
            "deepeval_labeled_rows": sum(1 for r in scored_rows if r["deepeval_success"] is not None),
            "deepeval_pass_rate": _mean([1.0 if bool(r["deepeval_success"]) else 0.0 for r in scored_rows if r["deepeval_success"] is not None]),
            "nli_entail_mean": _mean([float(r["nli_entail"]) for r in scored_rows]),
            "nli_neutral_mean": _mean([float(r["nli_neutral"]) for r in scored_rows]),
            "nli_contradict_mean": _mean([float(r["nli_contradict"]) for r in scored_rows]),
            "nli_argmax_entail_rate": _mean([1.0 if r["nli_argmax_label"] == "entail" else 0.0 for r in scored_rows]),
            "nli_argmax_neutral_rate": _mean([1.0 if r["nli_argmax_label"] == "neutral" else 0.0 for r in scored_rows]),
            "nli_argmax_contradict_rate": _mean([1.0 if r["nli_argmax_label"] == "contradict" else 0.0 for r in scored_rows]),
            "nli_model_name": args.nli_model_name,
        }
        write_csv([summary], args.out_csv)
        print(f"Saved summary: {args.out_csv}")


if __name__ == "__main__":
    main()
