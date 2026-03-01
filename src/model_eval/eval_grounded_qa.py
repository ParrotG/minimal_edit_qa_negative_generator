from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

from dataio import write_json, write_jsonl
from qa_checks import (
    CorrectnessConfig,
    check_answer_correctness,
    check_evidence_quotes,
    check_protocol_constraints,
    evaluate_structured_semantics,
)
from qa_judge.config import JudgeConfig, NLIConfig
from qa_judge.structured import StructuredAnswerJudge
from qa_protocol import parse_structured_output, validate_structured_payload

from .common import load_dataset_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_path", type=str, required=True, help="Generated structured outputs JSONL or dataset path.")
    parser.add_argument("--split", type=str, default="train", help="Split name when generated_path is a DatasetDict.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum evaluated rows. -1 means all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_semantics", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--metrics_out", type=str, required=True, help="Summary metrics JSON output path.")
    parser.add_argument("--details_out", type=str, default=None, help="Optional per-sample details JSONL path.")
    return parser.parse_args()


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _load_rows(generated_path: str, split: str, max_samples: int, seed: int) -> List[Dict[str, Any]]:
    ds = load_dataset_split(data_path=generated_path, split=split).shuffle(seed=seed)
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        out.append(dict(row))
        if max_samples > 0 and len(out) >= max_samples:
            break
    return out


def _build_structured_judge(args: argparse.Namespace) -> Optional[StructuredAnswerJudge]:
    if not args.enable_semantics:
        return None
    return StructuredAnswerJudge.from_defaults(
        nli_config=NLIConfig(
            model_name=args.nli_model_name,
            device=args.nli_device,
            batch_size=args.nli_batch_size,
            max_length=args.nli_max_length,
            fp16=args.nli_fp16,
        ),
        judge_config=JudgeConfig(
            temperature=args.temperature,
            full_margin_threshold=args.full_margin_threshold,
            reject_margin_threshold=args.reject_margin_threshold,
            reject_band_half_width=args.reject_band_half_width,
            qa_fail_as_negative=bool(args.qa_fail_as_negative),
            qa_check_answer_type=bool(args.qa_check_answer_type),
            qa_spacy_model=args.qa_spacy_model,
        ),
    )


def main() -> None:
    args = parse_args()
    rows = _load_rows(
        generated_path=args.generated_path,
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    if not rows:
        raise RuntimeError("No rows found for grounded-QA evaluation.")

    structured_judge = _build_structured_judge(args)
    correctness_cfg = CorrectnessConfig(
        semantic_match_f1_threshold=args.semantic_match_f1_threshold,
    )

    details: List[Dict[str, Any]] = []
    parse_ok = 0
    protocol_ok = 0
    evidence_ok = 0
    answerability_total = 0
    answerability_correct = 0
    correctness_total = 0
    correctness_correct = 0
    refusal_total = 0
    refusal_correct = 0
    confidence_totals = {"high": 0, "medium": 0, "low": 0}
    confidence_correct = {"high": 0, "medium": 0, "low": 0}

    for row in rows:
        parsed_output = row.get("parsed_output")
        parse_result = None
        if parsed_output:
            output = validate_structured_payload(parsed_output)
            parse_result = True
        else:
            result = parse_structured_output(str(row.get("raw_output") or row.get("answer") or ""))
            parse_result = result.ok
            output = result.parsed if result.ok else None

        record: Dict[str, Any] = {
            "model_tag": row.get("model_tag"),
            "model_step": row.get("model_step"),
            "model_path": row.get("model_path"),
            "source_id": row.get("source_id"),
            "sample_id": row.get("sample_id"),
            "parse_ok": bool(parse_result and output is not None),
        }

        sample_correct = False
        if parse_result and output is not None:
            parse_ok += 1
            protocol_report = check_protocol_constraints(output)
            evidence_report = check_evidence_quotes(output=output, knowledge=str(row.get("knowledge") or ""))
            semantics_report = evaluate_structured_semantics(
                question=str(row.get("question") or ""),
                knowledge=str(row.get("knowledge") or ""),
                output=output,
                judge=structured_judge,
                support_window_knowledge=str(row.get("knowledge") or ""),
            )

            protocol_ok += int(protocol_report.ok)
            evidence_ok += int(evidence_report.ok)

            gold_answerability = str(row.get("answerability_label") or "").strip()
            if gold_answerability:
                answerability_total += 1
                answerability_correct += int(output.answerability.value == gold_answerability)

            correctness_report = None
            if gold_answerability == "answerable" and str(row.get("reference_answer") or "").strip():
                correctness_total += 1
                correctness_report = check_answer_correctness(
                    answer=output.answer,
                    reference_answer=str(row.get("reference_answer") or ""),
                    cfg=correctness_cfg,
                )
                correctness_correct += int(correctness_report.ok)

            if gold_answerability == "unanswerable":
                refusal_total += 1
                refusal_ok = bool(protocol_report.ok)
                refusal_correct += int(refusal_ok)
                sample_correct = bool(output.answerability.value == gold_answerability and refusal_ok)
            else:
                sample_correct = bool(
                    protocol_report.ok
                    and evidence_report.ok
                    and (correctness_report.ok if correctness_report is not None else True)
                    and (semantics_report.ok is not False)
                    and (output.answerability.value == gold_answerability if gold_answerability else True)
                )

            confidence = output.confidence.value
            confidence_totals[confidence] += 1
            confidence_correct[confidence] += int(sample_correct)

            record.update(
                {
                    "parsed_output": output.model_dump(mode="json"),
                    "protocol_ok": protocol_report.ok,
                    "protocol_issues": protocol_report.issues,
                    "evidence_ok": evidence_report.ok,
                    "evidence_issues": evidence_report.issues,
                    "semantics_ok": semantics_report.ok,
                    "semantics_details": semantics_report.details,
                    "sample_correct": sample_correct,
                }
            )
            if correctness_report is not None:
                record["correctness_ok"] = correctness_report.ok
                record["correctness_exact_match"] = correctness_report.exact_match
                record["correctness_token_f1"] = correctness_report.token_f1
        details.append(record)

    confidence_accuracy = {
        key: _safe_rate(confidence_correct[key], confidence_totals[key])
        for key in ("high", "medium", "low")
    }
    metrics = {
        "num_rows": len(rows),
        "parse_ok_rate": _safe_rate(parse_ok, len(rows)),
        "protocol_ok_rate": _safe_rate(protocol_ok, parse_ok),
        "evidence_ok_rate": _safe_rate(evidence_ok, parse_ok),
        "answerability_accuracy": _safe_rate(answerability_correct, answerability_total),
        "answer_correct_rate": _safe_rate(correctness_correct, correctness_total),
        "refusal_correct_rate": _safe_rate(refusal_correct, refusal_total),
        "confidence_accuracy": confidence_accuracy,
        "confidence_totals": confidence_totals,
    }

    write_json(args.metrics_out, metrics)
    print(f"Saved metrics to: {args.metrics_out}")
    if args.details_out:
        write_jsonl(args.details_out, details)
        print(f"Saved details to: {args.details_out}")


if __name__ == "__main__":
    main()
