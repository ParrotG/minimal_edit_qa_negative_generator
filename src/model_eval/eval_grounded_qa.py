from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dataio import write_json, write_jsonl
from qa_checks import (
    CorrectnessConfig,
    SemanticCheckReport,
    check_answer_correctness,
    check_evidence_quotes,
    check_protocol_constraints,
    evaluate_structured_semantics_batch,
)
from qa_judge.config import JudgeConfig, NLIConfig
from qa_judge.structured import StructuredAnswerJudge
from qa_protocol import parse_structured_output, validate_structured_payload

from .common import (
    extract_knowledge_question_from_infer_prompt,
    infer_eval_track,
    infer_eval_variant,
    load_dataset_split,
    write_csv,
)
from .correctness_transformer_matcher import (
    AnswerEquivalenceTransformerMatcher,
    TransformerMatcherConfig,
    TransformerMatcherReviewReport,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_path", type=str, required=True, help="Generated structured outputs JSONL or dataset path.")
    parser.add_argument("--split", type=str, default="train", help="Split name when generated_path is a DatasetDict.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum evaluated samples by unique source/sample id. -1 means all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_semantics", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--matcher_threshold", type=float, default=TransformerMatcherConfig.threshold)
    parser.add_argument("--metrics_out", type=str, required=True, help="Per-model summary CSV path.")
    parser.add_argument("--details_out", type=str, default=None, help="Optional per-sample details JSONL path.")
    parser.add_argument("--confidence_out", type=str, default=None, help="Optional confidence correlation JSON path.")
    return parser.parse_args()


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _extract_model_key(row: Dict[str, Any]) -> Tuple[str, int, str, str, str]:
    return (
        str(row.get("model_tag") or row.get("tag") or "model"),
        int(row.get("model_step") or row.get("step") or 0),
        str(row.get("model_path") or row.get("model_name") or "model"),
        infer_eval_track(row),
        infer_eval_variant(row),
    )


def _extract_sample_key(row: Dict[str, Any]) -> str:
    source_id = str(row.get("source_id") or "").strip()
    if source_id:
        return source_id
    row_id = str(row.get("id") or "").strip()
    if row_id:
        return row_id
    sample_id = str(row.get("sample_id") or "").strip()
    if sample_id:
        return sample_id
    return str(hash(str(row)))


def _load_rows(generated_path: str, split: str, max_samples: int, seed: int) -> List[Dict[str, Any]]:
    ds = load_dataset_split(data_path=generated_path, split=split).shuffle(seed=seed)
    rows = [dict(row) for row in ds]
    if max_samples <= 0:
        return rows

    selected_keys: List[str] = []
    seen_keys: set[str] = set()
    for row in rows:
        key = _extract_sample_key(row)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected_keys.append(key)
        if len(selected_keys) >= max_samples:
            break

    selected = set(selected_keys)
    return [row for row in rows if _extract_sample_key(row) in selected]


def _resolve_question_knowledge(row: Dict[str, Any]) -> Tuple[str, str]:
    question = str(row.get("question") or "").strip()
    knowledge = str(row.get("knowledge") or "").strip()
    prompt = str(row.get("prompt") or "").strip()
    if question and knowledge:
        return question, knowledge
    if prompt:
        parsed_knowledge, parsed_question = extract_knowledge_question_from_infer_prompt(prompt)
        if not question:
            question = parsed_question
        if not knowledge:
            knowledge = parsed_knowledge
    return question.strip(), knowledge.strip()


def _extract_gold_answerability(row: Dict[str, Any]) -> str:
    label = str(row.get("answerability_label") or "").strip()
    if label:
        return label
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("answerability_label") or "").strip()
    return ""


def _extract_reference_answer(row: Dict[str, Any]) -> str:
    text = str(row.get("reference_answer") or "").strip()
    if text:
        return text
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("reference_answer") or "").strip()
    return ""


def _extract_confidence_rank(parsed_payload: Optional[Dict[str, Any]]) -> Optional[int]:
    if not parsed_payload:
        return None
    value = str(parsed_payload.get("confidence") or "").strip().lower()
    mapping = {"low": 1, "medium": 2, "high": 3}
    return mapping.get(value)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    den = den_x * den_y
    if den <= 0:
        return None
    return float(num / den)


def _rank_values(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[start][1]:
            end += 1
        avg_rank = (start + end) / 2.0 + 1.0
        for pos in range(start, end + 1):
            ranks[indexed[pos][0]] = avg_rank
        start = end + 1
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_rank_values(xs), _rank_values(ys))


def _normalized_resolution(bucket_values: Dict[str, List[float]]) -> Tuple[float, int]:
    """Compute Murphy normalized resolution for binary outcomes over confidence buckets."""

    total = sum(len(values) for values in bucket_values.values())
    if total <= 0:
        return float("nan"), 0
    all_values = [value for values in bucket_values.values() for value in values]
    mean_y = sum(all_values) / len(all_values)
    uncertainty = mean_y * (1.0 - mean_y)
    if uncertainty <= 0.0:
        return float("nan"), len(all_values)
    resolution = 0.0
    for values in bucket_values.values():
        if not values:
            continue
        bucket_rate = sum(values) / len(values)
        weight = len(values) / total
        resolution += weight * ((bucket_rate - mean_y) ** 2)
    return float(resolution / uncertainty), len(all_values)


def _empty_confidence_bucket(*, include_missing: bool = False) -> Dict[str, Dict[str, List[float]]]:
    labels = ["high", "medium", "low"]
    if include_missing:
        labels.append("missing")
    return {
        label: {
            "answerability": [],
            "correctness": [],
            "correctness_reviewed": [],
            "semantic_yes": [],
            "semantic_margin": [],
        }
        for label in labels
    }


def _empty_semantic_report() -> SemanticCheckReport:
    return SemanticCheckReport(
        ok=None,
        supported=None,
        answer_type_ok=None,
        refusal_ok=None,
        decision=None,
        margin=None,
        details={},
        issues=[],
    )


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


def _build_transformer_matcher(args: argparse.Namespace) -> AnswerEquivalenceTransformerMatcher:
    return AnswerEquivalenceTransformerMatcher(
        TransformerMatcherConfig(
            model_name=args.matcher_model_name,
            threshold=args.matcher_threshold,
        )
    )


def _evaluate_one_model(
    *,
    rows: Sequence[Dict[str, Any]],
    structured_judge: Optional[StructuredAnswerJudge],
    matcher: AnswerEquivalenceTransformerMatcher,
    correctness_cfg: CorrectnessConfig,
    semantic_decision_source: str,
    enable_semantics: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    semantic_inputs: List[Dict[str, Any]] = []
    semantic_prepared_indices: List[int] = []
    matcher_inputs: List[Dict[str, str]] = []
    matcher_prepared_indices: List[int] = []

    for row in rows:
        raw_text = str(row.get("raw_output") or row.get("answer") or "").strip()
        parsed_output_payload = row.get("parsed_output")
        output = None
        parse_errors: List[str] = []
        if isinstance(parsed_output_payload, dict):
            try:
                output = validate_structured_payload(parsed_output_payload)
            except Exception as exc:  # pragma: no cover
                parse_errors = [str(exc)]
        else:
            parse_result = parse_structured_output(raw_text)
            if parse_result.ok and parse_result.parsed is not None:
                output = parse_result.parsed
            else:
                parse_errors = list(parse_result.errors)

        question, knowledge = _resolve_question_knowledge(row)
        gold_answerability = _extract_gold_answerability(row)
        reference_answer = _extract_reference_answer(row)

        protocol_report = None
        evidence_report = None
        correctness_report = None
        should_run_semantics = False
        if output is not None:
            protocol_report = check_protocol_constraints(output)
            if output.answerability.value == "answerable":
                evidence_report = check_evidence_quotes(output=output, knowledge=knowledge)
                if reference_answer:
                    correctness_report = check_answer_correctness(
                        answer=output.answer,
                        reference_answer=reference_answer,
                        cfg=correctness_cfg,
                    )
                    if not correctness_report.ok:
                        matcher_inputs.append(
                            {
                                "question": question,
                                "reference_answer": reference_answer,
                                "answer": output.answer,
                            }
                        )
                        matcher_prepared_indices.append(len(prepared))
                should_run_semantics = bool(
                    enable_semantics and protocol_report.ok and len(output.evidence) > 0 and bool(question)
                )
                if should_run_semantics:
                    semantic_inputs.append(
                        {
                            "question": question,
                            "knowledge": knowledge,
                            "parsed_output": output.model_dump(mode="json"),
                        }
                    )
                    semantic_prepared_indices.append(len(prepared))

        prepared.append(
            {
                "row": row,
                "question": question,
                "knowledge": knowledge,
                "gold_answerability": gold_answerability,
                "reference_answer": reference_answer,
                "output": output,
                "parse_errors": parse_errors,
                "protocol_report": protocol_report,
                "evidence_report": evidence_report,
                "correctness_report": correctness_report,
                "correctness_matcher_report": None,
                "should_run_semantics": should_run_semantics,
                "semantics_report": _empty_semantic_report(),
            }
        )

    if matcher_inputs:
        matcher_reports = matcher.review_batch(matcher_inputs)
        for matcher_idx, prepared_idx in enumerate(matcher_prepared_indices):
            prepared[prepared_idx]["correctness_matcher_report"] = matcher_reports[matcher_idx]

    if semantic_inputs and enable_semantics:
        semantic_reports = evaluate_structured_semantics_batch(
            rows=semantic_inputs,
            judge=structured_judge,
            decision_source=semantic_decision_source,
        )
        for semantic_idx, prepared_idx in enumerate(semantic_prepared_indices):
            prepared[prepared_idx]["semantics_report"] = semantic_reports[semantic_idx]

    num_rows = len(prepared)
    num_parse_ok = 0
    num_protocol_ok = 0
    answerability_correct = 0
    predicted_answerable_total = 0
    evidence_ok = 0
    correctness_strict_ok = 0
    correctness_matcher_review_total = 0
    correctness_matcher_positive = 0
    correctness_reviewed_ok = 0
    semantic_evaluated_total = 0
    semantic_yes = 0
    semantic_margins: List[float] = []
    generation_failed_count = 0
    attempt_total = 0
    success_on_first_attempt = 0

    confidence_pairs_answerability_x: List[float] = []
    confidence_pairs_answerability_y: List[float] = []
    confidence_pairs_correctness_x: List[float] = []
    confidence_pairs_correctness_y: List[float] = []
    confidence_pairs_correctness_reviewed_x: List[float] = []
    confidence_pairs_correctness_reviewed_y: List[float] = []
    confidence_pairs_semantic_yes_x: List[float] = []
    confidence_pairs_semantic_yes_y: List[float] = []
    confidence_pairs_semantic_margin_x: List[float] = []
    confidence_pairs_semantic_margin_y: List[float] = []
    confidence_bucket_gated = _empty_confidence_bucket()
    confidence_bucket_all = _empty_confidence_bucket(include_missing=True)

    details: List[Dict[str, Any]] = []
    for item in prepared:
        row = item["row"]
        output = item["output"]
        parse_ok = bool(output is not None)
        protocol_report = item["protocol_report"]
        evidence_report = item["evidence_report"]
        correctness_report = item["correctness_report"]
        correctness_matcher_report: Optional[TransformerMatcherReviewReport] = item["correctness_matcher_report"]
        semantics_report: SemanticCheckReport = item["semantics_report"]
        gold_answerability = item["gold_answerability"]
        attempt_count = int(row.get("attempt_count") or 1)
        protocol_passed_from_generation = bool(row.get("protocol_passed")) if "protocol_passed" in row else bool(
            protocol_report.ok if protocol_report is not None else False
        )
        generation_failed = bool(row.get("generation_failed")) if "generation_failed" in row else (not protocol_passed_from_generation)
        attempt_total += max(1, attempt_count)
        generation_failed_count += int(generation_failed)
        if protocol_passed_from_generation and attempt_count == 1:
            success_on_first_attempt += 1

        if parse_ok:
            num_parse_ok += 1
            num_protocol_ok += int(bool(protocol_report and protocol_report.ok))

        pred_answerability = "" if output is None else output.answerability.value
        is_pred_answerable = pred_answerability == "answerable"

        answerability_match = None
        if parse_ok and gold_answerability:
            answerability_match = pred_answerability == gold_answerability
            answerability_correct += int(bool(answerability_match))

        reviewed_ok: Optional[bool] = None
        if parse_ok and is_pred_answerable:
            predicted_answerable_total += 1
            if evidence_report is not None:
                evidence_ok += int(evidence_report.ok)
            if correctness_report is not None:
                correctness_strict_ok += int(correctness_report.ok)
                reviewed_ok = bool(correctness_report.ok)
                if not correctness_report.ok and correctness_matcher_report is not None:
                    correctness_matcher_review_total += 1
                    correctness_matcher_positive += int(bool(correctness_matcher_report.ok))
                    reviewed_ok = bool(correctness_matcher_report.ok)
                correctness_reviewed_ok += int(bool(reviewed_ok))
            if enable_semantics and item["should_run_semantics"] and semantics_report.decision is not None:
                semantic_evaluated_total += 1
                semantic_yes += int(semantics_report.decision == "yes")
                if semantics_report.margin is not None:
                    semantic_margins.append(float(semantics_report.margin))

        parsed_payload = None if output is None else output.model_dump(mode="json")
        confidence_rank = _extract_confidence_rank(parsed_payload)
        confidence_label = "" if output is None else output.confidence.value
        confidence_bucket_label = confidence_label if confidence_label in confidence_bucket_all else "missing"

        answerability_all_value = 1.0 if bool(answerability_match) else 0.0
        correctness_strict_all_value = 1.0 if correctness_report is not None and correctness_report.ok else 0.0
        correctness_reviewed_all_value = 1.0 if bool(reviewed_ok) else 0.0
        semantic_yes_all_value = 1.0 if semantics_report.decision == "yes" else 0.0
        if semantics_report.margin is not None:
            confidence_bucket_all[confidence_bucket_label]["semantic_margin"].append(float(semantics_report.margin))
        confidence_bucket_all[confidence_bucket_label]["answerability"].append(answerability_all_value)
        confidence_bucket_all[confidence_bucket_label]["correctness"].append(correctness_strict_all_value)
        confidence_bucket_all[confidence_bucket_label]["correctness_reviewed"].append(correctness_reviewed_all_value)
        confidence_bucket_all[confidence_bucket_label]["semantic_yes"].append(semantic_yes_all_value)

        if confidence_rank is not None and answerability_match is not None:
            confidence_pairs_answerability_x.append(float(confidence_rank))
            confidence_pairs_answerability_y.append(1.0 if answerability_match else 0.0)
            if confidence_label in confidence_bucket_gated:
                confidence_bucket_gated[confidence_label]["answerability"].append(1.0 if answerability_match else 0.0)

        if parse_ok and is_pred_answerable and confidence_rank is not None:
            if correctness_report is not None:
                correctness_value = 1.0 if correctness_report.ok else 0.0
                confidence_pairs_correctness_x.append(float(confidence_rank))
                confidence_pairs_correctness_y.append(correctness_value)
                if confidence_label in confidence_bucket_gated:
                    confidence_bucket_gated[confidence_label]["correctness"].append(correctness_value)
                if reviewed_ok is not None:
                    reviewed_value = 1.0 if reviewed_ok else 0.0
                    confidence_pairs_correctness_reviewed_x.append(float(confidence_rank))
                    confidence_pairs_correctness_reviewed_y.append(reviewed_value)
                    if confidence_label in confidence_bucket_gated:
                        confidence_bucket_gated[confidence_label]["correctness_reviewed"].append(reviewed_value)
            if enable_semantics and item["should_run_semantics"] and semantics_report.decision in {"yes", "no"}:
                semantic_yes_value = 1.0 if semantics_report.decision == "yes" else 0.0
                confidence_pairs_semantic_yes_x.append(float(confidence_rank))
                confidence_pairs_semantic_yes_y.append(semantic_yes_value)
                if confidence_label in confidence_bucket_gated:
                    confidence_bucket_gated[confidence_label]["semantic_yes"].append(semantic_yes_value)
            if enable_semantics and item["should_run_semantics"] and semantics_report.margin is not None:
                margin_value = float(semantics_report.margin)
                confidence_pairs_semantic_margin_x.append(float(confidence_rank))
                confidence_pairs_semantic_margin_y.append(margin_value)
                if confidence_label in confidence_bucket_gated:
                    confidence_bucket_gated[confidence_label]["semantic_margin"].append(margin_value)

        details.append(
            {
                "model_tag": row.get("model_tag"),
                "model_step": row.get("model_step"),
                "model_path": row.get("model_path"),
                "eval_track": infer_eval_track(row),
                "eval_variant": infer_eval_variant(row),
                "sample_id": row.get("sample_id"),
                "source_id": row.get("source_id"),
                "id": row.get("id"),
                "data_split": row.get("data_split"),
                "answerability_label": gold_answerability,
                "parse_ok": parse_ok,
                "parse_errors": item["parse_errors"],
                "attempt_count": attempt_count,
                "protocol_passed_from_generation": protocol_passed_from_generation,
                "generation_failed": generation_failed,
                "pred_answerability": pred_answerability,
                "answerability_match": answerability_match,
                "protocol_ok": None if protocol_report is None else protocol_report.ok,
                "protocol_issues": [] if protocol_report is None else list(protocol_report.issues),
                "evidence_ok": None if evidence_report is None else evidence_report.ok,
                "evidence_issues": [] if evidence_report is None else list(evidence_report.issues),
                "correctness_ok": None if correctness_report is None else correctness_report.ok,
                "correctness_exact_match": None if correctness_report is None else correctness_report.exact_match,
                "correctness_token_f1": None if correctness_report is None else correctness_report.token_f1,
                "correctness_matcher_used": None if correctness_matcher_report is None else correctness_matcher_report.used,
                "correctness_matcher_ok": None if correctness_matcher_report is None else correctness_matcher_report.ok,
                "correctness_matcher_score": None if correctness_matcher_report is None else correctness_matcher_report.match_score,
                "correctness_matcher_issues": [] if correctness_matcher_report is None else list(correctness_matcher_report.issues),
                "correctness_reviewed_ok": reviewed_ok,
                "semantic_ok": semantics_report.ok if enable_semantics else None,
                "semantic_decision": semantics_report.decision if enable_semantics else None,
                "semantic_margin": semantics_report.margin if enable_semantics else None,
                "semantic_issues": list(semantics_report.issues) if enable_semantics else [],
                "confidence": confidence_label,
                "question": item["question"],
                "knowledge": item["knowledge"],
                "reference_answer": item["reference_answer"],
                "parsed_output": parsed_payload,
            }
        )

    metrics = {
        "num_rows": num_rows,
        "num_parse_ok": num_parse_ok,
        "parse_ok_rate": _safe_rate(num_parse_ok, num_rows),
        "num_protocol_ok": num_protocol_ok,
        "protocol_ok_rate_given_parse_ok": _safe_rate(num_protocol_ok, num_parse_ok),
        "protocol_ok_rate_all_samples": _safe_rate(num_protocol_ok, num_rows),
        "answerability_total": num_parse_ok,
        "answerability_correct": answerability_correct,
        "answerability_accuracy_given_parse_ok": _safe_rate(answerability_correct, num_parse_ok),
        "answerability_accuracy_all_samples": _safe_rate(answerability_correct, num_rows),
        "predicted_answerable_total": predicted_answerable_total,
        "evidence_total": predicted_answerable_total,
        "evidence_substring_ok": evidence_ok,
        "evidence_substring_ok_rate_on_pred_answerable": _safe_rate(evidence_ok, predicted_answerable_total),
        "evidence_substring_ok_rate_all_samples": _safe_rate(evidence_ok, num_rows),
        "correctness_total": predicted_answerable_total,
        "correctness_strict_ok": correctness_strict_ok,
        "correctness_strict_rate_on_pred_answerable": _safe_rate(correctness_strict_ok, predicted_answerable_total),
        "correctness_strict_rate_all_samples": _safe_rate(correctness_strict_ok, num_rows),
        "correctness_matcher_review_total": correctness_matcher_review_total,
        "correctness_matcher_positive": correctness_matcher_positive,
        "correctness_matcher_positive_rate": _safe_rate(correctness_matcher_positive, correctness_matcher_review_total),
        "correctness_reviewed_ok": correctness_reviewed_ok,
        "correctness_reviewed_rate_on_pred_answerable": _safe_rate(correctness_reviewed_ok, predicted_answerable_total),
        "correctness_reviewed_rate_all_samples": _safe_rate(correctness_reviewed_ok, num_rows),
        "semantic_total": semantic_evaluated_total,
        "semantic_yes": semantic_yes,
        "semantic_yes_rate_on_pred_answerable": _safe_rate(semantic_yes, predicted_answerable_total),
        "semantic_yes_rate_all_samples": _safe_rate(semantic_yes, num_rows),
        "semantic_margin_mean": _safe_mean(semantic_margins),
        "generation_failed_count": generation_failed_count,
        "generation_failed_rate": _safe_rate(generation_failed_count, num_rows),
        "avg_attempt_count": float(attempt_total / num_rows) if num_rows > 0 else float("nan"),
        "success_on_first_attempt_rate": _safe_rate(success_on_first_attempt, num_rows),
        "supporting_fact_check_enabled": False,
    }

    answerability_resolution_gated, answerability_resolution_gated_count = _normalized_resolution(
        {label: values["answerability"] for label, values in confidence_bucket_gated.items()}
    )
    answerability_resolution_all, answerability_resolution_all_count = _normalized_resolution(
        {label: values["answerability"] for label, values in confidence_bucket_all.items()}
    )
    correctness_reviewed_resolution_gated, correctness_reviewed_resolution_gated_count = _normalized_resolution(
        {label: values["correctness_reviewed"] for label, values in confidence_bucket_gated.items()}
    )
    correctness_reviewed_resolution_all, correctness_reviewed_resolution_all_count = _normalized_resolution(
        {label: values["correctness_reviewed"] for label, values in confidence_bucket_all.items()}
    )
    semantic_yes_resolution_gated, semantic_yes_resolution_gated_count = _normalized_resolution(
        {label: values["semantic_yes"] for label, values in confidence_bucket_gated.items()}
    )
    semantic_yes_resolution_all, semantic_yes_resolution_all_count = _normalized_resolution(
        {label: values["semantic_yes"] for label, values in confidence_bucket_all.items()}
    )

    metrics.update(
        {
            "confidence_resolution_answerability_given_parse_ok": answerability_resolution_gated,
            "confidence_resolution_answerability_given_parse_ok_num_rows": answerability_resolution_gated_count,
            "confidence_resolution_answerability_all_samples": answerability_resolution_all,
            "confidence_resolution_answerability_all_samples_num_rows": answerability_resolution_all_count,
            "confidence_resolution_correctness_reviewed_on_pred_answerable": correctness_reviewed_resolution_gated,
            "confidence_resolution_correctness_reviewed_on_pred_answerable_num_rows": correctness_reviewed_resolution_gated_count,
            "confidence_resolution_correctness_reviewed_all_samples": correctness_reviewed_resolution_all,
            "confidence_resolution_correctness_reviewed_all_samples_num_rows": correctness_reviewed_resolution_all_count,
            "confidence_resolution_semantic_yes_on_pred_answerable": semantic_yes_resolution_gated,
            "confidence_resolution_semantic_yes_on_pred_answerable_num_rows": semantic_yes_resolution_gated_count,
            "confidence_resolution_semantic_yes_all_samples": semantic_yes_resolution_all,
            "confidence_resolution_semantic_yes_all_samples_num_rows": semantic_yes_resolution_all_count,
        }
    )

    confidence_metrics = {
        "correlations": {
            "answerability": {
                "pearson": _pearson(confidence_pairs_answerability_x, confidence_pairs_answerability_y),
                "spearman": _spearman(confidence_pairs_answerability_x, confidence_pairs_answerability_y),
                "count": len(confidence_pairs_answerability_x),
            },
            "correctness_strict_on_pred_answerable": {
                "pearson": _pearson(confidence_pairs_correctness_x, confidence_pairs_correctness_y),
                "spearman": _spearman(confidence_pairs_correctness_x, confidence_pairs_correctness_y),
                "count": len(confidence_pairs_correctness_x),
            },
            "correctness_reviewed_on_pred_answerable": {
                "pearson": _pearson(confidence_pairs_correctness_reviewed_x, confidence_pairs_correctness_reviewed_y),
                "spearman": _spearman(confidence_pairs_correctness_reviewed_x, confidence_pairs_correctness_reviewed_y),
                "count": len(confidence_pairs_correctness_reviewed_x),
            },
            "semantic_yes_on_pred_answerable": {
                "pearson": _pearson(confidence_pairs_semantic_yes_x, confidence_pairs_semantic_yes_y),
                "spearman": _spearman(confidence_pairs_semantic_yes_x, confidence_pairs_semantic_yes_y),
                "count": len(confidence_pairs_semantic_yes_x),
            },
            "semantic_margin_on_pred_answerable": {
                "pearson": _pearson(confidence_pairs_semantic_margin_x, confidence_pairs_semantic_margin_y),
                "spearman": _spearman(confidence_pairs_semantic_margin_x, confidence_pairs_semantic_margin_y),
                "count": len(confidence_pairs_semantic_margin_x),
            },
        },
        "buckets_given_parse_ok_or_pred_answerable": {
            label: {
                "count": int(len(values["answerability"])),
                "answerability_accuracy_given_parse_ok": _safe_mean(values["answerability"]),
                "correctness_strict_rate_on_pred_answerable": _safe_mean(values["correctness"]),
                "correctness_reviewed_rate_on_pred_answerable": _safe_mean(values["correctness_reviewed"]),
                "semantic_yes_rate_on_pred_answerable": _safe_mean(values["semantic_yes"]),
                "semantic_margin_mean_on_pred_answerable": _safe_mean(values["semantic_margin"]),
            }
            for label, values in confidence_bucket_gated.items()
        },
        "buckets_all_samples": {
            label: {
                "count": int(len(values["answerability"])),
                "answerability_accuracy_all_samples": _safe_mean(values["answerability"]),
                "correctness_strict_rate_all_samples": _safe_mean(values["correctness"]),
                "correctness_reviewed_rate_all_samples": _safe_mean(values["correctness_reviewed"]),
                "semantic_yes_rate_all_samples": _safe_mean(values["semantic_yes"]),
                "semantic_margin_mean_observed": _safe_mean(values["semantic_margin"]),
            }
            for label, values in confidence_bucket_all.items()
        },
        "resolution": {
            "answerability_given_parse_ok": {
                "normalized_resolution": answerability_resolution_gated,
                "eligible_count": answerability_resolution_gated_count,
            },
            "answerability_all_samples": {
                "normalized_resolution": answerability_resolution_all,
                "eligible_count": answerability_resolution_all_count,
            },
            "correctness_reviewed_on_pred_answerable": {
                "normalized_resolution": correctness_reviewed_resolution_gated,
                "eligible_count": correctness_reviewed_resolution_gated_count,
            },
            "correctness_reviewed_all_samples": {
                "normalized_resolution": correctness_reviewed_resolution_all,
                "eligible_count": correctness_reviewed_resolution_all_count,
            },
            "semantic_yes_on_pred_answerable": {
                "normalized_resolution": semantic_yes_resolution_gated,
                "eligible_count": semantic_yes_resolution_gated_count,
            },
            "semantic_yes_all_samples": {
                "normalized_resolution": semantic_yes_resolution_all,
                "eligible_count": semantic_yes_resolution_all_count,
            },
        },
    }
    return metrics, details, confidence_metrics


def run_grounded_qa_evaluation(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run grounded-QA evaluation and return summary, details, and confidence rows."""

    rows = _load_rows(
        generated_path=args.generated_path,
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    if not rows:
        raise RuntimeError("No rows found for grounded-QA evaluation.")

    grouped: Dict[Tuple[str, int, str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_extract_model_key(row), []).append(row)

    structured_judge = _build_structured_judge(args)
    matcher = _build_transformer_matcher(args)
    correctness_cfg = CorrectnessConfig(semantic_match_f1_threshold=args.semantic_match_f1_threshold)

    summary_rows: List[Dict[str, Any]] = []
    details_rows: List[Dict[str, Any]] = []
    confidence_rows: List[Dict[str, Any]] = []

    for model_key, model_rows in sorted(grouped.items(), key=lambda item: (item[0][3], item[0][1], item[0][4], item[0][0], item[0][2])):
        metrics, details, confidence_metrics = _evaluate_one_model(
            rows=model_rows,
            structured_judge=structured_judge,
            matcher=matcher,
            correctness_cfg=correctness_cfg,
            semantic_decision_source=args.semantic_decision_source,
            enable_semantics=bool(args.enable_semantics),
        )
        summary_rows.append(
            {
                "model_tag": model_key[0],
                "model_step": model_key[1],
                "model_path": model_key[2],
                "eval_track": model_key[3],
                "eval_variant": model_key[4],
                **metrics,
            }
        )
        details_rows.extend(details)
        confidence_rows.append(
            {
                "model_tag": model_key[0],
                "model_step": model_key[1],
                "model_path": model_key[2],
                "eval_track": model_key[3],
                "eval_variant": model_key[4],
                **confidence_metrics,
            }
        )

    return summary_rows, details_rows, confidence_rows


def main() -> None:
    args = parse_args()
    summary_rows, details_rows, confidence_rows = run_grounded_qa_evaluation(args)

    write_csv(summary_rows, args.metrics_out)
    print(f"Saved metrics to: {args.metrics_out}")
    if args.details_out:
        write_jsonl(args.details_out, details_rows)
        print(f"Saved details to: {args.details_out}")
    if args.confidence_out:
        write_json(args.confidence_out, {"models": confidence_rows})
        print(f"Saved confidence analysis to: {args.confidence_out}")


if __name__ == "__main__":
    main()
