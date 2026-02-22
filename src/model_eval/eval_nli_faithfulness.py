from __future__ import annotations

import argparse
from typing import Any, Dict, List

try:
    from qa_judge.config import JudgeConfig, NLIConfig
    from qa_judge.judge import AnswerJudge
    from qa_judge.nli import NLIVerifier
except ImportError:  # pragma: no cover - compatibility fallback for editable installs.
    from qa_judge.config import JudgeConfig, NLIConfig
    from qa_judge.judge import AnswerJudge
    from qa_judge.nli import NLIVerifier

from dataio import write_jsonl
from .common import group_rows_by_model, load_generated_rows, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--generated_path", type=str, required=True, help="Generated answers JSONL or dataset path.")
    parser.add_argument("--split", type=str, default="train", help="Split name when generated_path is a DatasetDict.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum evaluated rows. -1 means all.")
    parser.add_argument("--seed", type=int, default=42)

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
    parser.add_argument("--temperature", type=float, default=JudgeConfig.temperature)
    parser.add_argument("--full_margin_threshold", type=float, default=JudgeConfig.full_margin_threshold)
    parser.add_argument("--reject_margin_threshold", type=float, default=JudgeConfig.reject_margin_threshold)
    parser.add_argument("--reject_band_half_width", type=float, default=JudgeConfig.reject_band_half_width)
    parser.add_argument("--qa_fail_as_negative", action=argparse.BooleanOptionalAction, default=JudgeConfig.qa_fail_as_negative)
    parser.add_argument("--qa_check_answer_type", action=argparse.BooleanOptionalAction, default=JudgeConfig.qa_check_answer_type)
    parser.add_argument("--qa_spacy_model", type=str, default=JudgeConfig.qa_spacy_model)

    parser.add_argument("--out_csv", type=str, default="nli_faithfulness_curve.csv")
    parser.add_argument("--out_jsonl", type=str, default="nli_faithfulness_samples.jsonl")
    return parser.parse_args()


def _build_answer_judge(args: argparse.Namespace) -> AnswerJudge:
    verifier = NLIVerifier(
        model_name=args.nli_model_name,
        device=args.nli_device,
        batch_size=args.nli_batch_size,
        max_length=args.nli_max_length,
        fp16=args.nli_fp16,
    )

    cfg = JudgeConfig(
        temperature=args.temperature,
        full_margin_threshold=args.full_margin_threshold,
        reject_margin_threshold=args.reject_margin_threshold,
        reject_band_half_width=args.reject_band_half_width,
        qa_fail_as_negative=bool(args.qa_fail_as_negative),
        qa_check_answer_type=bool(args.qa_check_answer_type),
        qa_spacy_model=args.qa_spacy_model,
    )
    return AnswerJudge(cfg=cfg, verifier=verifier)


def _attach_judgment(rows: List[Dict[str, Any]], judge: AnswerJudge) -> List[Dict[str, Any]]:
    if not rows:
        return []

    judge_inputs: List[Dict[str, Any]] = []
    for row in rows:
        judge_inputs.append(
            {
                "knowledge": row["knowledge"],
                "question": row["question"],
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


def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, str]:
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
    full_yes = [1.0 if (j.get("full_binary") or {}).get("decision") == "yes" else 0.0 for j in js]
    margins = [float(j.get("margin", 0.0)) for j in js]
    abstain = [1.0 if bool(j.get("is_abstain", False)) else 0.0 for j in js]

    qa_valid_values: List[float] = []
    for j in js:
        qa = (j.get("qa_consistency") or {}).get("answer_type_ok")
        if qa is None:
            continue
        qa_valid_values.append(1.0 if bool(qa) else 0.0)

    out.update(
        {
            "tag": tag,
            "step": str(step),
            "model_path": model_path,
            "num_samples": str(n),
            "nli_correct_rate": f"{_safe_mean(is_correct)}",
            "full_binary_yes_rate": f"{_safe_mean(full_yes)}",
            "margin_mean": f"{_safe_mean(margins)}",
            "reject_abstain_rate": f"{_safe_mean(abstain)}",
            "qa_answer_type_ok_rate": "" if not qa_valid_values else f"{_safe_mean(qa_valid_values)}",
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
        curve_rows.append(_summarize_rows(judged_rows))

    write_csv(curve_rows, args.out_csv)
    print(f"Saved: {args.out_csv}")

    write_jsonl(args.out_jsonl, all_rows)
    print(f"Saved: {args.out_jsonl}")


if __name__ == "__main__":
    main()
