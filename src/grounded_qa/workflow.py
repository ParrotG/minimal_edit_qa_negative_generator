from __future__ import annotations

import os
from dataclasses import asdict, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dataio import read_jsonl, write_json, write_jsonl
from llm_textgen.api_client import OpenAICompatibleTextGenerator
from qa_checks import (
    CorrectnessConfig,
    check_answer_correctness,
    check_evidence_against_supporting_facts,
    check_evidence_quotes,
    check_protocol_constraints,
    derive_confidence_label,
    evaluate_structured_semantics,
    score_validation_report,
    select_best_candidates,
)
from qa_checks.report import CorrectnessCheckReport, EvidenceCheckReport, ProtocolCheckReport, ValidationReport
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
    parse_structured_output,
    to_canonical_json,
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
    for row in rows:
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
                "raw_output": raw_output,
            }
        )

    metrics = {
        "num_examples": len(rows),
        "num_candidates": len(candidate_rows),
    }
    write_jsonl(out_path, candidate_rows)
    if metrics_out:
        write_json(metrics_out, metrics)
    return metrics


def _derive_support_window_knowledge(row: Dict[str, Any]) -> str:
    windows = [str(item.get("window_text") or "").strip() for item in list(row.get("supporting_sentences") or [])]
    windows = [item for item in windows if item]
    if windows:
        return "\n\n".join(windows)
    return str(row.get("knowledge") or "").strip()


def _coerce_confidence(
    report: ValidationReport,
    parse_ok: bool,
    parsed_output: Any,
) -> Tuple[str | None, Any]:
    if not parse_ok or parsed_output is None:
        return None, parsed_output
    derived = derive_confidence_label(report)
    if derived is None:
        return None, parsed_output
    updated = parsed_output.model_copy(update={"confidence": ConfidenceLevel(derived)})
    return derived, updated


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

    rows = list(read_jsonl(in_path))
    structured_judge = StructuredAnswerJudge.from_defaults() if validation_cfg.enable_semantics else None

    annotated_rows: List[Dict[str, Any]] = []
    num_parse_ok = 0
    num_overall_ok = 0
    answerability_match_yes = 0

    for row in rows:
        parse_result = parse_structured_output(str(row.get("raw_output") or ""), spec=spec)
        protocol_report = ProtocolCheckReport(ok=False, issues=["Parsing failed."])
        evidence_report = EvidenceCheckReport(ok=False, issues=["Parsing failed."])
        correctness_report: CorrectnessCheckReport | None = None
        semantics_report = None
        answerability_match = None

        if parse_result.ok and parse_result.parsed is not None:
            num_parse_ok += 1
            protocol_report = check_protocol_constraints(parse_result.parsed, spec=spec)
            evidence_report = check_evidence_quotes(
                output=parse_result.parsed,
                knowledge=str(row.get("knowledge") or ""),
                spec=spec,
            )
            support_evidence_report = check_evidence_against_supporting_facts(
                output=parse_result.parsed,
                supporting_sentences=list(row.get("supporting_sentences") or []),
            )
            evidence_report.ok = bool(evidence_report.ok and support_evidence_report.ok)
            evidence_report.issues.extend(support_evidence_report.issues)
            gold_answerability = str(row.get("answerability_label") or "").strip()
            if gold_answerability:
                answerability_match = parse_result.parsed.answerability.value == gold_answerability
                answerability_match_yes += int(answerability_match)

            if parse_result.parsed.answerability == Answerability.ANSWERABLE and str(row.get("reference_answer") or "").strip():
                correctness_report = check_answer_correctness(
                    answer=parse_result.parsed.answer,
                    reference_answer=str(row.get("reference_answer") or ""),
                    cfg=correctness_cfg,
                )

            semantics_report = evaluate_structured_semantics(
                question=str(row.get("question") or ""),
                knowledge=str(row.get("knowledge") or ""),
                output=parse_result.parsed,
                judge=structured_judge,
                support_window_knowledge=_derive_support_window_knowledge(row) if validation_cfg.use_support_window_knowledge else str(row.get("knowledge") or ""),
            )

        issues = list(parse_result.errors)
        issues.extend(protocol_report.issues)
        issues.extend(evidence_report.issues)
        if correctness_report is not None:
            issues.extend(correctness_report.issues)
        if semantics_report is not None:
            issues.extend(semantics_report.issues)

        overall_ok = bool(parse_result.ok and protocol_report.ok and evidence_report.ok)
        if answerability_match is False:
            overall_ok = False
        if str(row.get("answerability_label") or "").strip() == "answerable" and correctness_report is not None:
            overall_ok = bool(overall_ok and correctness_report.ok)
        if semantics_report is not None and semantics_report.ok is False:
            overall_ok = False

        report = ValidationReport(
            parse_ok=parse_result.ok,
            overall_ok=overall_ok,
            protocol=protocol_report,
            evidence=evidence_report,
            correctness=correctness_report,
            semantics=semantics_report,
            answerability_match=answerability_match,
            derived_confidence=None,
            selection_score=0.0,
            issues=issues,
        )
        derived_confidence, updated_output = _coerce_confidence(
            report=report,
            parse_ok=parse_result.ok,
            parsed_output=parse_result.parsed,
        )
        report.derived_confidence = derived_confidence
        report.selection_score = score_validation_report(report)

        canonical_output = None
        parsed_payload = None
        if updated_output is not None:
            parsed_payload = updated_output.model_dump(mode="json")
            canonical_output = to_canonical_json(updated_output, spec=spec)

        num_overall_ok += int(overall_ok)
        annotated_rows.append(
            {
                **row,
                "parse_ok": bool(parse_result.ok),
                "parsed_output": parsed_payload,
                "canonical_output": canonical_output,
                "validation_report": asdict(report),
            }
        )

    metrics = {
        "num_rows": len(rows),
        "num_parse_ok": num_parse_ok,
        "num_overall_ok": num_overall_ok,
        "num_answerability_match": answerability_match_yes,
    }
    write_jsonl(out_path, annotated_rows)
    if selected_out_path:
        write_jsonl(selected_out_path, select_best_candidates(annotated_rows))
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
