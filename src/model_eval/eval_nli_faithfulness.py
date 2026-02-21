from __future__ import annotations

import argparse
from typing import Any, Dict, List

try:
    from nli_judge.config import JudgeConfig, NLIConfig
    from nli_judge.judge import AnswerJudge
    from nli_judge.nli import NLIVerifier
except ImportError:  # pragma: no cover - compatibility fallback for editable installs.
    from nli_judge.config import JudgeConfig, NLIConfig
    from nli_judge.judge import AnswerJudge
    from nli_judge.nli import NLIVerifier

from dataio import write_jsonl
from .common import group_rows_by_model, load_generated_rows, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Input generated answers
    parser.add_argument("--generated_path", type=str, required=True, help="Generated answers JSONL or dataset path.")
    parser.add_argument("--split", type=str, default="train", help="Split name when generated_path is a DatasetDict.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum evaluated rows. -1 means all.")
    parser.add_argument("--seed", type=int, default=42)

    # NLI judgment
    parser.add_argument("--nli_model_name", type=str, default=NLIConfig.model_name)
    parser.add_argument("--secondary_nli_model_name", type=str, default=None)
    parser.add_argument("--nli_device", type=str, default=NLIConfig.device)
    parser.add_argument("--nli_batch_size", type=int, default=NLIConfig.batch_size)
    parser.add_argument("--nli_max_length", type=int, default=NLIConfig.max_length)
    parser.add_argument(
        "--nli_fp16",
        action=argparse.BooleanOptionalAction,
        default=NLIConfig.fp16,
        help="Whether to enable fp16 for NLI scoring on CUDA.",
    )
    parser.add_argument("--reference_entail_threshold", type=float, default=JudgeConfig.reference_entail_threshold)
    parser.add_argument("--candidate_entail_threshold", type=float, default=JudgeConfig.candidate_entail_threshold)
    parser.add_argument("--candidate_contradict_threshold", type=float, default=JudgeConfig.candidate_contradict_threshold)
    parser.add_argument("--vote_mode", type=str, default=JudgeConfig.vote_mode, choices=["primary", "and", "or"])
    parser.add_argument(
        "--qa_similarity_model_name",
        type=str,
        default="",
        help="Sentence-Transformer model name for optional QA consistency check. Empty disables it.",
    )
    parser.add_argument("--qa_similarity_min", type=float, default=JudgeConfig.qa_similarity_min)
    parser.add_argument(
        "--contradict_rate_threshold",
        type=float,
        default=0.5,
        help="Threshold used only for reporting contradiction-rate statistics.",
    )

    # Output
    parser.add_argument("--out_csv", type=str, default="nli_faithfulness_curve.csv")
    parser.add_argument("--out_jsonl", type=str, default="nli_faithfulness_samples.jsonl")
    return parser.parse_args()


def _build_answer_judge(args: argparse.Namespace) -> AnswerJudge:
    """Build ssqpg-compatible AnswerJudge for NLI faithfulness evaluation."""

    primary = NLIVerifier(
        model_name=args.nli_model_name,
        device=args.nli_device,
        batch_size=args.nli_batch_size,
        max_length=args.nli_max_length,
        fp16=args.nli_fp16,
    )

    secondary = None
    if args.secondary_nli_model_name:
        secondary = NLIVerifier(
            model_name=args.secondary_nli_model_name,
            device=args.nli_device,
            batch_size=args.nli_batch_size,
            max_length=args.nli_max_length,
            fp16=args.nli_fp16,
        )

    qa_model_name = args.qa_similarity_model_name.strip() or None
    cfg = JudgeConfig(
        reference_entail_threshold=args.reference_entail_threshold,
        candidate_entail_threshold=args.candidate_entail_threshold,
        candidate_contradict_threshold=args.candidate_contradict_threshold,
        vote_mode=args.vote_mode,
        qa_similarity_model_name=qa_model_name,
        qa_similarity_min=args.qa_similarity_min,
    )
    return AnswerJudge(cfg=cfg, primary_verifier=primary, secondary_verifier=secondary)


def _attach_judgment(rows: List[Dict[str, Any]], judge: AnswerJudge) -> List[Dict[str, Any]]:
    """Run NLI judgment and attach payload to generated rows."""

    if not rows:
        return []

    judge_inputs: List[Dict[str, Any]] = []
    for row in rows:
        judge_inputs.append(
            {
                "knowledge": row["knowledge"],
                "question": row["question"],
                "reference_answer": row.get("reference_answer", ""),
                "answer": row["answer"],
            }
        )

    judged_core, _ = judge.judge(judge_inputs)
    out: List[Dict[str, Any]] = []
    for base, judged in zip(rows, judged_core):
        rec = dict(base)
        rec["judge"] = judged.get("judge", {})
        out.append(rec)
    return out


def _safe_mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _summarize_rows(rows: List[Dict[str, Any]], contradict_rate_threshold: float) -> Dict[str, str]:
    """Build one model-level metric row for output CSV."""

    out: Dict[str, str] = {}
    if not rows:
        out["num_samples"] = "0"
        return out

    tag = str(rows[0]["model_tag"])
    step = int(rows[0]["model_step"])
    model_path = str(rows[0]["model_path"])

    js = [r.get("judge", {}) for r in rows]
    n = len(js)

    is_correct = [1.0 if bool(j.get("is_correct", False)) else 0.0 for j in js]
    candidate_supported = [1.0 if bool(j.get("candidate_supported", False)) else 0.0 for j in js]
    reference_supported = [1.0 if bool(j.get("reference_supported_primary", False)) else 0.0 for j in js]
    entail = [float(j.get("candidate_entail_primary", 0.0)) for j in js]
    contradict = [float(j.get("candidate_contradict_primary", 0.0)) for j in js]
    contradict_rate = [1.0 if x >= contradict_rate_threshold else 0.0 for x in contradict]

    qa_valid_values: List[float] = []
    for j in js:
        if j.get("qa_similarity") is None:
            continue
        qa_valid_values.append(1.0 if bool(j.get("qa_consistent", False)) else 0.0)

    out.update(
        {
            "tag": tag,
            "step": str(step),
            "model_path": model_path,
            "num_samples": str(n),
            "nli_correct_rate": f"{_safe_mean(is_correct)}",
            "candidate_supported_rate": f"{_safe_mean(candidate_supported)}",
            "reference_supported_rate": f"{_safe_mean(reference_supported)}",
            "candidate_entail_mean": f"{_safe_mean(entail)}",
            "candidate_contradict_mean": f"{_safe_mean(contradict)}",
            "candidate_contradict_rate": f"{_safe_mean(contradict_rate)}",
            "candidate_contradict_rate_threshold": f"{contradict_rate_threshold}",
            "qa_consistent_rate": "" if not qa_valid_values else f"{_safe_mean(qa_valid_values)}",
        }
    )

    return out


def main() -> None:
    args = parse_args()

    rows = load_generated_rows(
        generated_path=args.generated_path,
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    if not rows:
        raise RuntimeError("No valid generated rows found for NLI faithfulness evaluation.")

    judge = _build_answer_judge(args)
    grouped = group_rows_by_model(rows)

    all_rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, str]] = []
    for _, model_rows in sorted(grouped.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        judged_rows = _attach_judgment(model_rows, judge=judge)
        all_rows.extend(judged_rows)
        curve_rows.append(
            _summarize_rows(
                judged_rows,
                contradict_rate_threshold=args.contradict_rate_threshold,
            )
        )

    write_csv(curve_rows, args.out_csv)
    print(f"Saved: {args.out_csv}")

    write_jsonl(all_rows, args.out_jsonl)
    print(f"Saved: {args.out_jsonl}")


if __name__ == "__main__":
    main()
