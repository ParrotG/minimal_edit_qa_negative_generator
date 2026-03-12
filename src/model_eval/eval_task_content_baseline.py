from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dataio import write_jsonl
from llm_textgen.api_client import OpenAICompatibleTextGenerator
from qa_checks import CorrectnessConfig, check_answer_correctness
from qa_judge.config import JudgeConfig, NLIConfig
from qa_judge.judge import AnswerJudge
from qa_judge.nli import NLIVerifier

from .answer_extraction import AnswerExtractionConfig, extract_answers_with_llm
from .common import (
    flat_content_eval_skipped_reason,
    group_rows_by_model,
    is_flat_answerable_for_content_eval,
    load_generated_rows,
    write_csv,
)
from .correctness_transformer_matcher import (
    AnswerEquivalenceTransformerMatcher,
    TransformerMatcherConfig,
    TransformerMatcherReviewReport,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_path", type=str, required=True, help="Generated answers JSONL or dataset path.")
    parser.add_argument("--split", type=str, default="train", help="Split name when generated_path is a DatasetDict.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum evaluated rows per run. -1 means all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--semantic_decision_source", type=str, default="full_binary", choices=["full_binary", "reject_aware"])
    parser.add_argument("--semantic_match_f1_threshold", type=float, default=0.85)
    parser.add_argument("--nli_model_name", type=str, default=NLIConfig.model_name)
    parser.add_argument("--nli_device", type=str, default=NLIConfig.device)
    parser.add_argument("--nli_batch_size", type=int, default=NLIConfig.batch_size)
    parser.add_argument("--nli_max_length", type=int, default=NLIConfig.max_length)
    parser.add_argument("--nli_fp16", action=argparse.BooleanOptionalAction, default=NLIConfig.fp16)
    parser.add_argument("--temperature", type=float, default=JudgeConfig.temperature)
    parser.add_argument("--full_margin_threshold", type=float, default=JudgeConfig.full_margin_threshold)
    parser.add_argument("--reject_margin_threshold", type=float, default=JudgeConfig.reject_margin_threshold)
    parser.add_argument("--reject_band_half_width", type=float, default=JudgeConfig.reject_band_half_width)
    parser.add_argument("--qa_fail_as_negative", action=argparse.BooleanOptionalAction, default=JudgeConfig.qa_fail_as_negative)
    parser.add_argument("--qa_check_answer_type", action=argparse.BooleanOptionalAction, default=JudgeConfig.qa_check_answer_type)
    parser.add_argument("--qa_spacy_model", type=str, default=JudgeConfig.qa_spacy_model)
    parser.add_argument("--matcher_model_name", type=str, default=TransformerMatcherConfig.model_name)
    parser.add_argument("--api_model_name", type=str, default=AnswerExtractionConfig.api_model_name)
    parser.add_argument("--api_base_url", type=str, default=AnswerExtractionConfig.api_base_url)
    parser.add_argument("--api_key_env", type=str, default=AnswerExtractionConfig.api_key_env)
    parser.add_argument("--api_timeout_seconds", type=float, default=AnswerExtractionConfig.api_timeout_seconds)
    parser.add_argument("--api_max_concurrency", type=int, default=AnswerExtractionConfig.api_max_concurrency)
    parser.add_argument("--api_max_retries", type=int, default=AnswerExtractionConfig.api_max_retries)
    parser.add_argument("--api_backoff_base_seconds", type=float, default=AnswerExtractionConfig.api_backoff_base_seconds)
    parser.add_argument("--api_backoff_max_seconds", type=float, default=AnswerExtractionConfig.api_backoff_max_seconds)
    parser.add_argument("--api_max_new_tokens", type=int, default=AnswerExtractionConfig.max_new_tokens)
    parser.add_argument("--api_temperature", type=float, default=AnswerExtractionConfig.temperature)
    parser.add_argument("--api_top_p", type=float, default=AnswerExtractionConfig.top_p)
    parser.add_argument("--api_seed", type=int, default=AnswerExtractionConfig.seed)
    parser.add_argument("--error_log_dir", type=str, default=AnswerExtractionConfig.error_log_dir)
    parser.add_argument("--extraction_max_attempts", type=int, default=AnswerExtractionConfig.max_attempts)
    parser.add_argument("--metrics_out", type=str, required=True, help="Per-model summary CSV path.")
    parser.add_argument("--details_out", type=str, default=None, help="Optional per-sample details JSONL path.")
    return parser.parse_args()


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _build_answer_judge(args: argparse.Namespace) -> AnswerJudge:
    verifier = NLIVerifier(
        model_name=args.nli_model_name,
        device=args.nli_device,
        batch_size=args.nli_batch_size,
        max_length=args.nli_max_length,
        fp16=bool(args.nli_fp16),
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


def _build_transformer_matcher(args: argparse.Namespace) -> AnswerEquivalenceTransformerMatcher:
    return AnswerEquivalenceTransformerMatcher(
        TransformerMatcherConfig(
            model_name=args.matcher_model_name,
        )
    )


def _semantic_decision(judge_payload: Dict[str, Any], decision_source: str) -> str:
    if decision_source == "reject_aware":
        return str(((judge_payload.get("reject_aware") or {}).get("decision")) or "abstain")
    return str(((judge_payload.get("full_binary") or {}).get("decision")) or "no")


def _build_extraction_cfg(args: argparse.Namespace) -> AnswerExtractionConfig:
    return AnswerExtractionConfig(
        api_model_name=args.api_model_name,
        api_base_url=args.api_base_url,
        api_key_env=args.api_key_env,
        api_timeout_seconds=args.api_timeout_seconds,
        api_max_concurrency=args.api_max_concurrency,
        api_max_retries=args.api_max_retries,
        api_backoff_base_seconds=args.api_backoff_base_seconds,
        api_backoff_max_seconds=args.api_backoff_max_seconds,
        max_new_tokens=args.api_max_new_tokens,
        temperature=args.api_temperature,
        top_p=args.api_top_p,
        seed=args.api_seed,
        error_log_dir=args.error_log_dir,
        max_attempts=args.extraction_max_attempts,
    )


def _evaluate_model_rows(
    *,
    rows: List[Dict[str, Any]],
    extractor: OpenAICompatibleTextGenerator,
    extraction_cfg: AnswerExtractionConfig,
    judge: AnswerJudge,
    matcher: AnswerEquivalenceTransformerMatcher,
    decision_source: str,
    correctness_cfg: CorrectnessConfig,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not rows:
        return {
            "num_rows": 0,
            "extraction_parse_ok_rate": float("nan"),
            "refusal_detected_rate": float("nan"),
            "answerability_total": 0,
            "answerability_accuracy": float("nan"),
            "content_eval_total": 0,
            "content_eval_rate": float("nan"),
            "num_refusal_or_unanswerable_skipped": 0,
            "correctness_total": 0,
            "correctness_strict_rate": float("nan"),
            "correctness_matcher_review_total": 0,
            "correctness_matcher_positive_rate": float("nan"),
            "correctness_reviewed_rate": float("nan"),
            "semantic_total": 0,
            "semantic_yes_rate": float("nan"),
            "semantic_margin_mean": float("nan"),
        }, []

    extracted_rows = extract_answers_with_llm(rows=rows, generator=extractor, cfg=extraction_cfg)

    matcher_inputs: List[Dict[str, str]] = []
    matcher_indices: List[int] = []
    judge_inputs: List[Dict[str, Any]] = []
    judge_indices: List[int] = []
    details: List[Dict[str, Any]] = []

    parse_ok_count = 0
    refusal_detected_count = 0
    answerability_total = 0
    answerability_correct = 0
    content_eval_total = 0
    refusal_or_unanswerable_skipped = 0
    correctness_total = 0
    correctness_strict_ok = 0

    for idx, row in enumerate(extracted_rows):
        extraction_parse_ok = bool(row.get("extraction_parse_ok"))
        parse_ok_count += int(extraction_parse_ok)
        refusal_detected = row.get("refusal_detected")
        if refusal_detected is True:
            refusal_detected_count += 1

        gold_label = str(row.get("answerability_label") or "").strip()
        pred_label: Optional[str] = None
        if extraction_parse_ok and refusal_detected is not None:
            pred_label = "unanswerable" if bool(refusal_detected) else "answerable"
            if gold_label:
                answerability_total += 1
                answerability_correct += int(pred_label == gold_label)

        extracted_answer = str(row.get("extracted_answer") or "").strip()
        gate_row = {
            **row,
            "pred_answerability": pred_label,
        }
        entered_content_eval = is_flat_answerable_for_content_eval(gate_row)
        skipped_reason = "" if entered_content_eval else flat_content_eval_skipped_reason(gate_row)
        if entered_content_eval:
            content_eval_total += 1
        elif skipped_reason in {"predicted_refusal", "predicted_unanswerable"}:
            refusal_or_unanswerable_skipped += 1
        strict_report = None
        if entered_content_eval and str(row.get("reference_answer") or "").strip():
            correctness_total += 1
            strict_report = check_answer_correctness(
                answer=extracted_answer,
                reference_answer=str(row.get("reference_answer") or "").strip(),
                cfg=correctness_cfg,
            )
            correctness_strict_ok += int(strict_report.ok)
            if not strict_report.ok:
                matcher_inputs.append(
                    {
                        "question": str(row.get("question") or ""),
                        "reference_answer": str(row.get("reference_answer") or "").strip(),
                        "answer": extracted_answer,
                    }
                )
                matcher_indices.append(idx)
        if entered_content_eval:
            judge_inputs.append(
                {
                    "knowledge": str(row.get("knowledge") or ""),
                    "question": str(row.get("question") or ""),
                    "answer": extracted_answer,
                }
            )
            judge_indices.append(idx)

        details.append(
            {
                **row,
                "pred_answerability": pred_label,
                "answerability_match": None if pred_label is None or not gold_label else bool(pred_label == gold_label),
                "strict_report": strict_report,
                "correctness_matcher_report": None,
                "judge_payload": None,
                "semantic_decision": None,
                "semantic_margin": None,
                "entered_content_eval": entered_content_eval,
                "content_eval_skipped_reason": skipped_reason,
            }
        )

    if matcher_inputs:
        matcher_reports = matcher.review_batch(matcher_inputs)
        for matcher_idx, detail_idx in enumerate(matcher_indices):
            details[detail_idx]["correctness_matcher_report"] = matcher_reports[matcher_idx]

    if judge_inputs:
        judged_rows, _ = judge.judge(judge_inputs)
        for judged_idx, detail_idx in enumerate(judge_indices):
            judge_payload = dict((judged_rows[judged_idx].get("judge") or {}))
            details[detail_idx]["judge_payload"] = judge_payload
            details[detail_idx]["semantic_decision"] = _semantic_decision(judge_payload, decision_source=decision_source)
            details[detail_idx]["semantic_margin"] = judge_payload.get("margin")

    correctness_matcher_review_total = 0
    correctness_matcher_positive = 0
    correctness_reviewed_ok = 0
    semantic_total = 0
    semantic_yes = 0
    semantic_margins: List[float] = []
    output_rows: List[Dict[str, Any]] = []

    for row in details:
        strict_report = row["strict_report"]
        matcher_report: Optional[TransformerMatcherReviewReport] = row["correctness_matcher_report"]
        reviewed_ok = None
        if strict_report is not None:
            reviewed_ok = bool(strict_report.ok)
            if not strict_report.ok and matcher_report is not None:
                correctness_matcher_review_total += 1
                correctness_matcher_positive += int(bool(matcher_report.ok))
                reviewed_ok = bool(matcher_report.ok)
            correctness_reviewed_ok += int(bool(reviewed_ok))

        semantic_decision = row["semantic_decision"]
        semantic_margin = row["semantic_margin"]
        if semantic_decision in {"yes", "no"}:
            semantic_total += 1
            semantic_yes += int(semantic_decision == "yes")
        if semantic_margin is not None:
            semantic_margins.append(float(semantic_margin))

        output_rows.append(
            {
                "model_tag": row.get("model_tag"),
                "model_step": row.get("model_step"),
                "model_path": row.get("model_path"),
                "eval_track": row.get("eval_track"),
                "eval_variant": row.get("eval_variant"),
                "sample_id": row.get("sample_id"),
                "source_id": row.get("source_id"),
                "data_split": row.get("data_split"),
                "question": row.get("question"),
                "reference_answer": row.get("reference_answer"),
                "raw_answer": row.get("answer"),
                "answerability_label": row.get("answerability_label"),
                "pred_answerability": row.get("pred_answerability"),
                "answerability_match": row.get("answerability_match"),
                "extraction_raw_response": row.get("extraction_raw_response"),
                "extraction_attempt_count": row.get("extraction_attempt_count"),
                "extraction_api_ok": row.get("extraction_api_ok"),
                "extraction_parse_ok": row.get("extraction_parse_ok"),
                "extraction_error_type": row.get("extraction_error_type"),
                "extraction_error_message": row.get("extraction_error_message"),
                "extraction_errors": row.get("extraction_errors"),
                "extracted_answer": row.get("extracted_answer"),
                "refusal_detected": row.get("refusal_detected"),
                "refusal_reason": row.get("refusal_reason"),
                "entered_content_eval": bool(row.get("entered_content_eval")),
                "content_eval_skipped_reason": row.get("content_eval_skipped_reason"),
                "correctness_ok": None if strict_report is None else strict_report.ok,
                "correctness_exact_match": None if strict_report is None else strict_report.exact_match,
                "correctness_token_f1": None if strict_report is None else strict_report.token_f1,
                "correctness_matcher_used": None if matcher_report is None else matcher_report.used,
                "correctness_matcher_ok": None if matcher_report is None else matcher_report.ok,
                "correctness_matcher_score": None if matcher_report is None else matcher_report.match_score,
                "correctness_reviewed_ok": reviewed_ok,
                "semantic_decision": semantic_decision,
                "semantic_margin": semantic_margin,
            }
        )

    metrics = {
        "num_rows": len(rows),
        "extraction_parse_ok_rate": _safe_rate(parse_ok_count, len(rows)),
        "refusal_detected_rate": _safe_rate(refusal_detected_count, len(rows)),
        "answerability_total": answerability_total,
        "answerability_correct": answerability_correct,
        "answerability_accuracy": _safe_rate(answerability_correct, answerability_total),
        "content_eval_total": content_eval_total,
        "content_eval_rate": _safe_rate(content_eval_total, len(rows)),
        "num_refusal_or_unanswerable_skipped": refusal_or_unanswerable_skipped,
        "correctness_total": correctness_total,
        "correctness_strict_ok": correctness_strict_ok,
        "correctness_strict_rate": _safe_rate(correctness_strict_ok, correctness_total),
        "correctness_matcher_review_total": correctness_matcher_review_total,
        "correctness_matcher_positive": correctness_matcher_positive,
        "correctness_matcher_positive_rate": _safe_rate(correctness_matcher_positive, correctness_matcher_review_total),
        "correctness_reviewed_ok": correctness_reviewed_ok,
        "correctness_reviewed_rate": _safe_rate(correctness_reviewed_ok, correctness_total),
        "semantic_total": semantic_total,
        "semantic_yes_rate": _safe_rate(semantic_yes, semantic_total),
        "semantic_margin_mean": _safe_mean(semantic_margins),
    }
    return metrics, output_rows


def run_task_content_baseline(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate base task generations and return summary and detail rows."""

    rows = load_generated_rows(
        generated_path=args.generated_path,
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
        answer_source="answer",
    )
    if not rows:
        raise RuntimeError("No valid generated rows found for task-content baseline evaluation.")

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Environment variable {args.api_key_env} is not set.")

    grouped = group_rows_by_model(rows)
    extraction_cfg = _build_extraction_cfg(args)
    extractor = OpenAICompatibleTextGenerator(cfg=extraction_cfg.to_api_config(), api_key=api_key)
    judge = _build_answer_judge(args)
    matcher = _build_transformer_matcher(args)
    correctness_cfg = CorrectnessConfig(semantic_match_f1_threshold=args.semantic_match_f1_threshold)

    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    for (model_tag, model_step, model_path, eval_track, eval_variant), model_rows in sorted(
        grouped.items(),
        key=lambda item: (item[0][3], item[0][1], item[0][4], item[0][0], item[0][2]),
    ):
        metrics, details = _evaluate_model_rows(
            rows=model_rows,
            extractor=extractor,
            extraction_cfg=extraction_cfg,
            judge=judge,
            matcher=matcher,
            decision_source=args.semantic_decision_source,
            correctness_cfg=correctness_cfg,
        )
        summary_rows.append(
            {
                "model_tag": model_tag,
                "model_step": model_step,
                "model_path": model_path,
                "eval_track": eval_track,
                "eval_variant": eval_variant,
                **metrics,
            }
        )
        detail_rows.extend(details)

    return summary_rows, detail_rows


def main() -> None:
    args = parse_args()
    summary_rows, detail_rows = run_task_content_baseline(args)

    write_csv(summary_rows, args.metrics_out)
    print(f"Saved metrics to: {args.metrics_out}")
    if args.details_out:
        write_jsonl(args.details_out, detail_rows)
        print(f"Saved details to: {args.details_out}")


if __name__ == "__main__":
    main()
