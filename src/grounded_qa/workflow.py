from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from dataio import read_jsonl, write_json, write_jsonl
from llm_textgen.api_client import OpenAICompatibleTextGenerator
from qa_checks import (
    CorrectnessConfig,
    check_answer_correctness,
    check_evidence_against_supporting_facts,
    check_evidence_quotes,
    check_protocol_constraints,
    evaluate_structured_semantics,
    score_validation_report,
    select_best_candidates,
)
from qa_checks.report import CorrectnessCheckReport, EvidenceCheckReport, ProtocolCheckReport, SemanticCheckReport, ValidationReport
from qa_checks.selection import assign_quantile_confidence_labels
from qa_data import (
    ConstructionConfig,
    HotpotSourceConfig,
    NegativeSamplingConfig,
    SplitConfig,
    assign_split,
    build_answerable_example,
    derive_simple_unanswerable,
    example_from_dict,
    example_to_dict,
    iter_hotpot_rows,
)
from qa_judge.structured import StructuredAnswerJudge
from qa_protocol import (
    Answerability,
    ConfidenceLevel,
    DEFAULT_PROTOCOL_SPEC,
    build_infer_prompt,
    build_teacher_prompt,
    count_text_tokens,
    parse_structured_output,
    to_canonical_json,
    validate_structured_payload,
)
from qa_protocol.spec import ProtocolSpec
from sft_trainer.formatting import build_sft_record

from .config import TeacherGenerationConfig, ValidationConfig


def build_hotpot_source(
    *,
    out_path: str,
    metrics_out: Optional[str],
    source_cfg: HotpotSourceConfig,
    construct_cfg: ConstructionConfig,
) -> Dict[str, int]:
    """Build answerable source examples from HotpotQA."""

    rows: List[Dict[str, Any]] = []
    num_input = 0
    num_kept = 0

    for raw_row in iter_hotpot_rows(source_cfg):
        num_input += 1
        example = build_answerable_example(raw_row, construct_cfg)
        if example is None:
            continue
        num_kept += 1
        rows.append(example_to_dict(example))

    metrics = {
        "num_input": num_input,
        "num_kept": num_kept,
        "num_dropped": num_input - num_kept,
    }
    write_jsonl(out_path, rows)
    if metrics_out:
        write_json(metrics_out, metrics)
    return metrics


def build_simple_negatives(
    *,
    in_path: str,
    out_path: str,
    metrics_out: Optional[str],
    negative_cfg: NegativeSamplingConfig,
) -> Dict[str, int]:
    """Build first-pass unanswerable rows from answerable examples."""

    source_rows = list(read_jsonl(in_path))
    out_rows: List[Dict[str, Any]] = []
    num_kept = 0

    for row in source_rows:
        example = example_from_dict(dict(row))
        derived = derive_simple_unanswerable(example, negative_cfg)
        if derived is None:
            continue
        num_kept += 1
        out_rows.append(example_to_dict(derived))

    metrics = {
        "num_input": len(source_rows),
        "num_kept": num_kept,
        "num_dropped": len(source_rows) - num_kept,
    }
    write_jsonl(out_path, out_rows)
    if metrics_out:
        write_json(metrics_out, metrics)
    return metrics


def split_examples(
    *,
    in_path: str,
    out_path: str,
    metrics_out: Optional[str],
    split_cfg: SplitConfig,
) -> Dict[str, Any]:
    """Assign stable split labels to raw examples."""

    rows = list(read_jsonl(in_path))
    out_rows: List[Dict[str, Any]] = []
    split_counts: Dict[str, int] = {}

    for row in rows:
        example = assign_split(example_from_dict(dict(row)), split_cfg)
        split_counts[example.split or ""] = split_counts.get(example.split or "", 0) + 1
        out_rows.append(example_to_dict(example))

    metrics: Dict[str, Any] = {
        "num_rows": len(out_rows),
        "split_counts": split_counts,
    }
    write_jsonl(out_path, out_rows)
    if metrics_out:
        write_json(metrics_out, metrics)
    return metrics


def _prompt_within_budget(row: Dict[str, Any], cfg: TeacherGenerationConfig, spec: ProtocolSpec) -> tuple[bool, int, str]:
    prompt = build_infer_prompt(
        knowledge=str(row.get("knowledge") or "").strip(),
        question=str(row.get("question") or "").strip(),
        spec=spec,
    )
    prompt_tokens = count_text_tokens(prompt, cfg.prefilter_tokenizer_name)
    return prompt_tokens <= cfg.max_prompt_tokens, prompt_tokens, prompt


def generate_teacher_candidates(
    *,
    in_path: str,
    out_path: str,
    metrics_out: Optional[str],
    cfg: TeacherGenerationConfig,
    spec: ProtocolSpec = DEFAULT_PROTOCOL_SPEC,
) -> Dict[str, int]:
    """Generate teacher candidates for structured grounded QA."""

    rows = list(read_jsonl(in_path))
    api_key = os.getenv(cfg.api_key_env, "").strip()
    generator = OpenAICompatibleTextGenerator(cfg=cfg.to_api_config(), api_key=api_key)

    prompts: List[str] = []
    jobs: List[Dict[str, Any]] = []
    num_drop_prompt_over_budget = 0
    num_retained_examples = 0

    for row in rows:
        within_budget, prompt_tokens, prefilter_prompt = _prompt_within_budget(row, cfg, spec)
        if not within_budget:
            num_drop_prompt_over_budget += 1
            continue

        num_retained_examples += 1
        for candidate_idx in range(cfg.num_candidates_per_example):
            prompt = build_teacher_prompt(
                knowledge=str(row.get("knowledge") or "").strip(),
                question=str(row.get("question") or "").strip(),
                reference_answer=str(row.get("reference_answer") or "").strip() or None,
                spec=spec,
            )
            jobs.append(
                {
                    "row": row,
                    "candidate_id": candidate_idx,
                    "prompt": prompt,
                    "prefilter_prompt": prefilter_prompt,
                    "prefilter_prompt_tokens": prompt_tokens,
                }
            )
            prompts.append(prompt)

    outputs = generator.generate_many(prompts)
    candidate_rows: List[Dict[str, Any]] = []
    for job, raw_output in zip(jobs, outputs):
        base = dict(job["row"])
        candidate_rows.append(
            {
                **base,
                "candidate_id": int(job["candidate_id"]),
                "teacher_model": cfg.api_model_name,
                "prompt_style": cfg.prompt_style,
                "prompt": job["prompt"],
                "prefilter_prompt": job["prefilter_prompt"],
                "prefilter_prompt_tokens": int(job["prefilter_prompt_tokens"]),
                "raw_output": raw_output,
            }
        )

    metrics = {
        "num_examples": len(rows),
        "num_retained_examples": num_retained_examples,
        "num_drop_prompt_over_budget": num_drop_prompt_over_budget,
        "num_candidates": len(candidate_rows),
    }
    write_jsonl(out_path, candidate_rows)
    if metrics_out:
        write_json(metrics_out, metrics)
    return metrics


def _semantic_hard_fail(decision: Optional[str], validation_cfg: ValidationConfig) -> bool:
    if not validation_cfg.semantic_drop_by_nli or decision is None:
        return False
    if validation_cfg.semantic_decision_source == "reject_aware":
        return decision in {"no", "abstain"}
    return decision == "no"


def _build_empty_semantics_report() -> SemanticCheckReport:
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


def _refresh_selected_confidence(
    *,
    rows: List[Dict[str, Any]],
    validation_cfg: ValidationConfig,
    spec: ProtocolSpec,
) -> None:
    assign_quantile_confidence_labels(rows)
    for row in rows:
        validation_report = dict(row.get("validation_report") or {})
        derived_confidence = validation_report.get("derived_confidence")
        parsed_output = row.get("parsed_output")
        if derived_confidence is None or not parsed_output:
            continue
        updated_output = validate_structured_payload(parsed_output).model_copy(
            update={"confidence": ConfidenceLevel(derived_confidence)}
        )
        canonical_output = to_canonical_json(updated_output, spec=spec)
        completion_tokens = count_text_tokens(canonical_output, validation_cfg.tokenizer_name)
        completion_over_budget = completion_tokens > validation_cfg.max_completion_tokens

        row["parsed_output"] = updated_output.model_dump(mode="json")
        row["canonical_output"] = canonical_output
        validation_report["derived_confidence"] = derived_confidence
        validation_report["completion_tokens"] = completion_tokens
        validation_report["completion_over_budget"] = completion_over_budget
        row["validation_report"] = validation_report


def validate_teacher_candidates(
    *,
    in_path: str,
    out_path: str,
    selected_out_path: Optional[str],
    metrics_out: Optional[str],
    validation_cfg: ValidationConfig,
    correctness_cfg: CorrectnessConfig = CorrectnessConfig(),
    spec: ProtocolSpec = DEFAULT_PROTOCOL_SPEC,
) -> Dict[str, Any]:
    """Validate teacher-generated structured outputs and optionally select the best candidate."""

    if validation_cfg.semantic_decision_source not in {"full_binary", "reject_aware"}:
        raise ValueError(f"Unsupported semantic_decision_source: {validation_cfg.semantic_decision_source}")

    rows = list(read_jsonl(in_path))
    structured_judge = StructuredAnswerJudge.from_defaults() if validation_cfg.enable_semantics else None

    annotated_rows: List[Dict[str, Any]] = []
    num_parse_ok = 0
    num_hard_pass = 0
    num_answerability_match = 0
    num_semantic_negative = 0
    num_completion_over_budget = 0

    for row in rows:
        parse_result = parse_structured_output(str(row.get("raw_output") or ""), spec=spec)
        protocol_report = ProtocolCheckReport(ok=False, issues=["Parsing failed."])
        evidence_report = EvidenceCheckReport(ok=False, issues=["Parsing failed."])
        support_evidence_report = EvidenceCheckReport(ok=True, issues=[])
        correctness_report: CorrectnessCheckReport | None = None
        semantics_report = _build_empty_semantics_report()
        answerability_match = None
        hard_fail_reasons: List[str] = list(parse_result.errors)
        soft_metrics: Dict[str, float | bool | None] = {}
        completion_tokens = None
        completion_over_budget = False
        parsed_payload = None
        canonical_output = None

        if parse_result.ok and parse_result.parsed is not None:
            num_parse_ok += 1
            protocol_report = check_protocol_constraints(parse_result.parsed, spec=spec)
            parsed_payload = parse_result.parsed.model_dump(mode="json")
            canonical_output = to_canonical_json(parse_result.parsed, spec=spec)
            completion_tokens = count_text_tokens(canonical_output, validation_cfg.tokenizer_name)
            completion_over_budget = completion_tokens > validation_cfg.max_completion_tokens

            evidence_report = check_evidence_quotes(
                output=parse_result.parsed,
                knowledge=str(row.get("knowledge") or ""),
                spec=spec,
            )
            support_evidence_report = check_evidence_against_supporting_facts(
                output=parse_result.parsed,
                supporting_sentences=list(row.get("supporting_sentences") or []),
            )
            evidence_report.supporting_fact_match_rate = support_evidence_report.supporting_fact_match_rate
            evidence_report.outside_supporting_fact_count = support_evidence_report.outside_supporting_fact_count

            gold_answerability = str(row.get("answerability_label") or "").strip()
            if gold_answerability:
                answerability_match = parse_result.parsed.answerability.value == gold_answerability
                num_answerability_match += int(bool(answerability_match))

            if parse_result.parsed.answerability == Answerability.ANSWERABLE and str(row.get("reference_answer") or "").strip():
                correctness_report = check_answer_correctness(
                    answer=parse_result.parsed.answer,
                    reference_answer=str(row.get("reference_answer") or ""),
                    cfg=correctness_cfg,
                )

            if validation_cfg.enable_semantics:
                semantics_report = evaluate_structured_semantics(
                    question=str(row.get("question") or ""),
                    knowledge=str(row.get("knowledge") or ""),
                    output=parse_result.parsed,
                    judge=structured_judge,
                    decision_source=validation_cfg.semantic_decision_source,
                )
                if semantics_report.decision == "no":
                    num_semantic_negative += 1

            soft_metrics = {
                "semantic_margin": semantics_report.margin,
                "supporting_fact_match_rate": evidence_report.supporting_fact_match_rate,
                "outside_supporting_fact_count": float(evidence_report.outside_supporting_fact_count),
                "correctness_token_f1": None if correctness_report is None else correctness_report.token_f1,
                "evidence_count": float(protocol_report.evidence_count),
            }

            if not protocol_report.ok:
                hard_fail_reasons.extend(protocol_report.issues)
            if completion_over_budget:
                num_completion_over_budget += 1
                hard_fail_reasons.append(
                    f"Canonical completion exceeds max_completion_tokens={validation_cfg.max_completion_tokens}."
                )
            if answerability_match is False:
                hard_fail_reasons.append("Predicted answerability does not match the target label.")
            if not evidence_report.ok:
                hard_fail_reasons.extend(evidence_report.issues)
            if correctness_report is not None and not correctness_report.ok:
                hard_fail_reasons.extend(correctness_report.issues)
            if _semantic_hard_fail(semantics_report.decision, validation_cfg):
                hard_fail_reasons.extend(semantics_report.issues or ["Semantic verifier rejected the sample."])

        issues = list(hard_fail_reasons)
        issues.extend(support_evidence_report.issues if parse_result.ok and parse_result.parsed is not None else [])
        if correctness_report is not None:
            for issue in correctness_report.issues:
                if issue not in issues:
                    issues.append(issue)
        for issue in semantics_report.issues:
            if issue not in issues:
                issues.append(issue)

        hard_pass = len(hard_fail_reasons) == 0
        num_hard_pass += int(hard_pass)
        report = ValidationReport(
            parse_ok=parse_result.ok,
            overall_ok=hard_pass,
            hard_pass=hard_pass,
            protocol=protocol_report,
            evidence=evidence_report,
            correctness=correctness_report,
            semantics=semantics_report if validation_cfg.enable_semantics else None,
            answerability_match=answerability_match,
            derived_confidence=None,
            selection_score=0.0,
            hard_fail_reasons=hard_fail_reasons,
            soft_metrics=soft_metrics,
            semantic_margin=semantics_report.margin if validation_cfg.enable_semantics else None,
            semantic_decision_source=validation_cfg.semantic_decision_source if validation_cfg.enable_semantics else None,
            semantic_filter_applied=bool(validation_cfg.enable_semantics and validation_cfg.semantic_drop_by_nli),
            completion_tokens=completion_tokens,
            completion_over_budget=completion_over_budget,
            issues=issues,
        )
        report.selection_score = score_validation_report(asdict(report))

        annotated_rows.append(
            {
                **row,
                "parse_ok": bool(parse_result.ok),
                "parsed_output": parsed_payload,
                "canonical_output": canonical_output,
                "validation_report": asdict(report),
            }
        )

    selected_rows = select_best_candidates(annotated_rows)
    _refresh_selected_confidence(rows=selected_rows, validation_cfg=validation_cfg, spec=spec)

    metrics = {
        "num_rows": len(rows),
        "num_parse_ok": num_parse_ok,
        "num_hard_pass": num_hard_pass,
        "num_answerability_match": num_answerability_match,
        "num_semantic_negative": num_semantic_negative,
        "num_completion_over_budget": num_completion_over_budget,
        "num_selected": len(selected_rows),
    }
    write_jsonl(out_path, annotated_rows)
    if selected_out_path:
        write_jsonl(selected_out_path, selected_rows)
    if metrics_out:
        write_json(metrics_out, metrics)
    return metrics


def build_sft_records(
    *,
    in_path: str,
    out_path: str,
    metrics_out: Optional[str],
    prompt_style: str,
    keep_only_overall_ok: bool,
) -> Dict[str, int]:
    """Pack selected validated rows into SFT prompt-completion records."""

    rows = list(read_jsonl(in_path))
    out_rows: List[Dict[str, Any]] = []

    for row in rows:
        validation_report = dict(row.get("validation_report") or {})
        if keep_only_overall_ok and not bool(validation_report.get("overall_ok")):
            continue
        record = build_sft_record(
            row=dict(row),
            prompt_style=prompt_style,
        )
        if record is None:
            continue
        out_rows.append(record)

    metrics = {
        "num_input": len(rows),
        "num_kept": len(out_rows),
        "num_dropped": len(rows) - len(out_rows),
    }
    write_jsonl(out_path, out_rows)
    if metrics_out:
        write_json(metrics_out, metrics)
    return metrics
