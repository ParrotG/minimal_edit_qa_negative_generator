from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from nli_judge.config import JudgeConfig, NLIConfig
    from nli_judge.judge import AnswerJudge
    from nli_judge.nli import NLIVerifier
except ImportError:  # pragma: no cover - compatibility fallback for editable installs.
    from nli_judge.config import JudgeConfig, NLIConfig
    from nli_judge.judge import AnswerJudge
    from nli_judge.nli import NLIVerifier

from dataio import write_jsonl
from .common import group_rows_by_model, load_dataset_split, normalize_list, safe_int, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Input: DeepEval details artifact.
    parser.add_argument(
        "--deepeval_details_path",
        type=str,
        required=True,
        help="DeepEval details path (JSONL or save_to_disk dataset).",
    )
    parser.add_argument("--split", type=str, default="train", help="Split name when details path is a DatasetDict.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum rows to compare. -1 means all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_error_rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip rows where DeepEval reported non-empty error.",
    )

    # Label semantics alignment.
    parser.add_argument(
        "--deepeval_success_means_supported",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True, DeepEval success=True is treated as faithful/supported.",
    )

    # NLI judgment settings.
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
    parser.add_argument("--candidate_entail_threshold", type=float, default=JudgeConfig.candidate_entail_threshold)
    parser.add_argument("--candidate_contradict_threshold", type=float, default=JudgeConfig.candidate_contradict_threshold)
    parser.add_argument("--vote_mode", type=str, default=JudgeConfig.vote_mode, choices=["primary", "and", "or"])

    # Outputs.
    parser.add_argument("--out_csv", type=str, default="nli_deepeval_consistency.csv")
    parser.add_argument("--out_jsonl", type=str, default="nli_deepeval_consistency_samples.jsonl")
    return parser.parse_args()


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _load_deepeval_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
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
            }
        )

        if args.max_samples > 0 and len(out) >= args.max_samples:
            break
    return out


def _build_answer_judge(args: argparse.Namespace) -> AnswerJudge:
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

    cfg = JudgeConfig(
        reference_entail_threshold=JudgeConfig.reference_entail_threshold,
        candidate_entail_threshold=args.candidate_entail_threshold,
        candidate_contradict_threshold=args.candidate_contradict_threshold,
        vote_mode=args.vote_mode,
        qa_similarity_model_name=None,
        qa_similarity_min=JudgeConfig.qa_similarity_min,
    )
    return AnswerJudge(cfg=cfg, primary_verifier=primary, secondary_verifier=secondary)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _binary_stats(preds: Sequence[bool], refs: Sequence[bool]) -> Dict[str, float]:
    n = len(preds)
    if n == 0:
        return {
            "accuracy": float("nan"),
            "agreement_rate": float("nan"),
            "cohen_kappa": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "tp": 0.0,
            "tn": 0.0,
            "fp": 0.0,
            "fn": 0.0,
        }

    tp = sum(1 for p, r in zip(preds, refs) if p and r)
    tn = sum(1 for p, r in zip(preds, refs) if (not p) and (not r))
    fp = sum(1 for p, r in zip(preds, refs) if p and (not r))
    fn = sum(1 for p, r in zip(preds, refs) if (not p) and r)

    accuracy = (tp + tn) / n
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)

    p_yes_pred = sum(1 for x in preds if x) / n
    p_yes_ref = sum(1 for x in refs if x) / n
    p_e = p_yes_pred * p_yes_ref + (1.0 - p_yes_pred) * (1.0 - p_yes_ref)
    kappa = float("nan") if abs(1.0 - p_e) < 1e-12 else (accuracy - p_e) / (1.0 - p_e)

    return {
        "accuracy": float(accuracy),
        "agreement_rate": float(accuracy),
        "cohen_kappa": float(kappa),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def _pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2 or len(ys) != n:
        return float("nan")

    mean_x = _mean(xs)
    mean_y = _mean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return float("nan")
    return float(cov / math.sqrt(var_x * var_y))


def _rankdata(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # one-based average rank
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    return _pearson_corr(_rankdata(xs), _rankdata(ys))


def _attach_nli(rows: List[Dict[str, Any]], judge: AnswerJudge) -> List[Dict[str, Any]]:
    judge_inputs = [
        {
            "knowledge": row["knowledge"],
            "question": row["question"],
            "reference_answer": "",
            "answer": row["answer"],
        }
        for row in rows
    ]
    judged_rows, _ = judge.judge(judge_inputs)

    out: List[Dict[str, Any]] = []
    for base, judged in zip(rows, judged_rows):
        judge_payload = judged.get("judge", {})
        nli_supported = bool(judge_payload.get("candidate_supported", False))
        deepeval_success = base.get("deepeval_success")

        out.append(
            {
                **base,
                "nli_supported": nli_supported,
                "nli_is_correct": bool(judge_payload.get("is_correct", False)),
                "nli_candidate_entail": _safe_float(judge_payload.get("candidate_entail_primary")),
                "nli_candidate_contradict": _safe_float(judge_payload.get("candidate_contradict_primary")),
                "nli_vote_mode": judge_payload.get("vote_mode"),
                "is_comparable": deepeval_success is not None,
                "label_agree": None if deepeval_success is None else bool(nli_supported == deepeval_success),
            }
        )
    return out


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"num_rows": 0}

    tag = str(rows[0]["model_tag"])
    step = int(rows[0]["model_step"])
    model_path = str(rows[0]["model_path"])

    comparable_rows = [r for r in rows if r.get("is_comparable")]
    preds = [bool(r["nli_supported"]) for r in comparable_rows]
    refs = [bool(r["deepeval_success"]) for r in comparable_rows]
    bstats = _binary_stats(preds, refs)

    scored_rows = [r for r in rows if r.get("deepeval_score") is not None]
    deepeval_scores = [float(r["deepeval_score"]) for r in scored_rows]
    nli_contradict = [float(r["nli_candidate_contradict"]) for r in scored_rows if r.get("nli_candidate_contradict") is not None]
    nli_entail = [float(r["nli_candidate_entail"]) for r in scored_rows if r.get("nli_candidate_entail") is not None]

    corr_rows_contra = [r for r in scored_rows if r.get("nli_candidate_contradict") is not None]
    score_for_contra = [float(r["deepeval_score"]) for r in corr_rows_contra]
    contra_values = [float(r["nli_candidate_contradict"]) for r in corr_rows_contra]

    corr_rows_entail = [r for r in scored_rows if r.get("nli_candidate_entail") is not None]
    score_for_entail = [float(r["deepeval_score"]) for r in corr_rows_entail]
    entail_values = [float(r["nli_candidate_entail"]) for r in corr_rows_entail]

    return {
        "model_tag": tag,
        "model_step": step,
        "model_path": model_path,
        "num_rows": len(rows),
        "num_comparable": len(comparable_rows),
        "num_scored": len(scored_rows),
        "deepeval_pass_rate": _mean([1.0 if bool(r.get("deepeval_success")) else 0.0 for r in comparable_rows]),
        "nli_supported_rate": _mean([1.0 if bool(r.get("nli_supported")) else 0.0 for r in rows]),
        "nli_accuracy_vs_deepeval": bstats["accuracy"],
        "agreement_rate": bstats["agreement_rate"],
        "cohen_kappa": bstats["cohen_kappa"],
        "faithful_precision": bstats["precision"],
        "faithful_recall": bstats["recall"],
        "faithful_f1": bstats["f1"],
        "tp": bstats["tp"],
        "tn": bstats["tn"],
        "fp": bstats["fp"],
        "fn": bstats["fn"],
        "corr_pearson_score_vs_nli_contradict": _pearson_corr(score_for_contra, contra_values),
        "corr_spearman_score_vs_nli_contradict": _spearman_corr(score_for_contra, contra_values),
        "corr_pearson_score_vs_nli_entail": _pearson_corr(score_for_entail, entail_values),
        "corr_spearman_score_vs_nli_entail": _spearman_corr(score_for_entail, entail_values),
        "deepeval_score_mean": _mean(deepeval_scores),
        "nli_contradict_mean": _mean(nli_contradict),
        "nli_entail_mean": _mean(nli_entail),
    }


def main() -> None:
    args = parse_args()

    base_rows = _load_deepeval_rows(args)
    if not base_rows:
        raise RuntimeError("No valid DeepEval detail rows found for consistency comparison.")

    judge = _build_answer_judge(args)
    grouped = group_rows_by_model(base_rows)

    sample_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for _, rows in sorted(grouped.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        compared_rows = _attach_nli(rows, judge=judge)
        sample_rows.extend(compared_rows)
        summary_rows.append(_summarize(compared_rows))

    write_csv(summary_rows, args.out_csv)
    print(f"Saved summary: {args.out_csv}")

    write_jsonl(sample_rows, args.out_jsonl)
    print(f"Saved per-sample comparison: {args.out_jsonl}")


if __name__ == "__main__":
    main()
