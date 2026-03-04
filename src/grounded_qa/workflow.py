from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dataio import read_jsonl, write_json, write_jsonl
from llm_textgen.api_client import OpenAICompatibleTextGenerator
from qa_checks import (
    CorrectnessConfig,
    check_answer_correctness,
    check_evidence_against_supporting_facts,
    check_evidence_quotes,
    check_protocol_constraints,
    evaluate_structured_semantics,
    prefilter_mixed_examples,
    score_validation_report,
    select_best_candidates,
    select_first_valid_unanswerable_candidate,
)
from qa_checks.report import CorrectnessCheckReport, EvidenceCheckReport, ProtocolCheckReport, SemanticCheckReport, ValidationReport
from qa_checks.selection import assign_quantile_confidence_labels
from qa_data import (
    AnswerabilitySplitConfig,
    ConstructionConfig,
    DataSplitConfig,
    HotpotSourceConfig,
    UnanswerableBuildConfig,
    build_prepared_examples,
    example_from_dict,
    example_to_dict,
    iter_tagged_hotpot_rows,
    write_partitioned_examples,
)
from qa_judge.config import JudgeConfig, NLIConfig
from qa_judge.judge import AnswerJudge
from qa_judge.nli import NLIVerifier
from qa_judge.structured import StructuredAnswerJudge
from qa_protocol import (
    Answerability,
    ConfidenceLevel,
    DEFAULT_PROTOCOL_SPEC,
    build_teacher_prompt,
    count_text_tokens,
    parse_structured_output,
    to_canonical_json,
    validate_structured_payload,
)
from qa_protocol.spec import ProtocolSpec
from sft_trainer.formatting import build_sft_record

from .config import SourcePrefilterConfig, TeacherGenerationConfig, ValidationConfig


def tag_source_rows(
    *,
    out_path: str,
    metrics_out: Optional[str],
    source_cfg: HotpotSourceConfig,
    data_split_cfg: DataSplitConfig,
    answerability_split_cfg: AnswerabilitySplitConfig,
) -> Dict[str, Any]:
    """Read Hotpot rows and assign independent data and answerability split tags."""

    rows = list(iter_tagged_hotpot_rows(source_cfg, data_split_cfg, answerability_split_cfg))
    data_split_counts: Dict[str, int] = {}
    answerability_split_counts: Dict[str, int] = {}
    for row in rows:
        data_split = str(row.get("data_split") or "").strip()
        answerability_split = str(row.get("answerability_split") or "").strip()
        data_split_counts[data_split] = data_split_counts.get(data_split, 0) + 1
        answerability_split_counts[answerability_split] = answerability_split_counts.get(answerability_split, 0) + 1

    metrics = {
        "num_rows": len(rows),
        "data_split_counts": data_split_counts,
        "answerability_split_counts": answerability_split_counts,
    }
    write_jsonl(out_path, rows)
    if metrics_out:
        write_json(metrics_out, metrics)
    return metrics


def build_source_examples(
    *,
    in_path: str,
    out_path: str,
    metrics_out: Optional[str],
    construct_cfg: ConstructionConfig,
    unanswerable_cfg: UnanswerableBuildConfig,
) -> Dict[str, Any]:
    """Build mixed concrete grounded-QA examples from tagged raw rows."""

    rows = [dict(row) for row in read_jsonl(in_path)]
    built_examples = build_prepared_examples(rows, construct_cfg, unanswerable_cfg)
    out_rows = [example_to_dict(example) for example in built_examples]

    by_label: Dict[str, int] = {}
    by_data_split: Dict[str, int] = {}
    by_answerability_split: Dict[str, int] = {}
    for row in out_rows:
        label = str(row.get("answerability_label") or "").strip()
        data_split = str(row.get("data_split") or "").strip()
        answerability_split = str(row.get("answerability_split") or "").strip()
        by_label[label] = by_label.get(label, 0) + 1
        by_data_split[data_split] = by_data_split.get(data_split, 0) + 1
        by_answerability_split[answerability_split] = by_answerability_split.get(answerability_split, 0) + 1

    metrics = {
        "num_input_rows": len(rows),
        "num_output_rows": len(out_rows),
        "answerability_label_counts": by_label,
        "data_split_counts": by_data_split,
        "answerability_split_counts": by_answerability_split,
    }
    write_jsonl(out_path, out_rows)
    if metrics_out:
        write_json(metrics_out, metrics)
    return metrics


def _build_source_prefilter_judge(cfg: SourcePrefilterConfig) -> AnswerJudge:
    verifier = NLIVerifier(
        model_name=cfg.nli_model_name,
        device=cfg.nli_device,
        batch_size=cfg.nli_batch_size,
        max_length=cfg.nli_max_length,
        fp16=cfg.nli_fp16,
    )
    judge_cfg = JudgeConfig(
        temperature=cfg.temperature,
        full_margin_threshold=cfg.full_margin_threshold,
        reject_margin_threshold=cfg.full_margin_threshold,
        reject_band_half_width=0.0,
        qa_fail_as_negative=False,
        qa_check_answer_type=False,
        qa_spacy_model=JudgeConfig.qa_spacy_model,
    )
    return AnswerJudge(cfg=judge_cfg, verifier=verifier)


def prefilter_source_examples(
    *,
    in_path: str,
    out_path: str,
    metrics_out: Optional[str],
    cfg: SourcePrefilterConfig,
) -> Dict[str, Any]:
    """Apply mixed source-stage filtering before teacher generation."""

    rows = [dict(row) for row in read_jsonl(in_path)]
    judge = _build_source_prefilter_judge(cfg) if cfg.enable_unanswerable_nli else None
    kept_rows, metrics = prefilter_mixed_examples(
        rows,
        tokenizer_name=cfg.tokenizer_name,
        max_prompt_tokens=cfg.max_prompt_tokens,
        enable_unanswerable_nli=cfg.enable_unanswerable_nli,
        judge=judge,
    )
    write_jsonl(out_path, kept_rows)
    if metrics_out:
        write_json(metrics_out, metrics)
    return metrics


def partition_source_examples(
    *,
    in_path: str,
    out_dir: str,
    metrics_out: Optional[str],
) -> Dict[str, Any]:
    """Partition mixed prepared examples by downstream data split."""

    rows = [dict(row) for row in read_jsonl(in_path)]
    os.makedirs(out_dir, exist_ok=True)
    metrics = write_partitioned_examples(rows, out_dir, metrics_out=metrics_out)
    return metrics


def generate_teacher_candidates(
    *,
    in_path: str,
    out_path: str,
    metrics_out: Optional[str],
    cfg: TeacherGenerationConfig,
    spec: ProtocolSpec = DEFAULT_PROTOCOL_SPEC,
) -> Dict[str, int]:
    """Generate teacher candidates for mixed structured grounded-QA examples."""

    rows = list(read_jsonl(in_path))
    api_key = os.getenv(cfg.api_key_env, "").strip()
    generator = OpenAICompatibleTextGenerator(cfg=cfg.to_api_config(), api_key=api_key)

    prompts: List[str] = []
    jobs: List[Dict[str, Any]] = []
    for row in rows:
        reference_answer = (
            str(row.get("reference_answer") or "").strip() or None
            if str(row.get("answerability_label") or "") == "answerable"
            else None
        )
        num_candidates = (
            cfg.answerable_num_candidates_per_example
            if str(row.get("answerability_label") or "") == "answerable"
            else cfg.unanswerable_num_candidates_per_example
        )
        for candidate_idx in range(num_candidates):
            prompt = build_teacher_prompt(
                knowledge=str(row.get("knowledge") or "").strip(),
                question=str(row.get("question") or "").strip(),
                reference_answer=reference_answer,
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
        "num_answerable_examples": sum(1 for row in rows if str(row.get("answerability_label") or "") == "answerable"),
        "num_unanswerable_examples": sum(1 for row in rows if str(row.get("answerability_label") or "") == "unanswerable"),
        "answerable_num_candidates_per_example": cfg.answerable_num_candidates_per_example,
        "unanswerable_num_candidates_per_example": cfg.unanswerable_num_candidates_per_example,
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


def _resolve_reverse_confidence_labels(count: int) -> list[str]:
    if count <= 0:
        return []
    if count == 1:
        return ["medium"]
    if count == 2:
        return ["high", "low"]

    low_cut = count // 3
    high_cut = low_cut
    middle_count = count - low_cut - high_cut
    labels: list[str] = []
    labels.extend(["high"] * high_cut)
    labels.extend(["medium"] * middle_count)
    labels.extend(["low"] * low_cut)
    return labels


def _refresh_selected_unanswerable_confidence(
    *,
    rows: List[Dict[str, Any]],
    validation_cfg: ValidationConfig,
    spec: ProtocolSpec,
) -> None:
    scored_rows: list[tuple[float, Dict[str, Any]]] = []
    for row in rows:
        parsed_output = row.get("parsed_output") or {}
        if str(parsed_output.get("answerability") or "") != "unanswerable":
            continue
        metadata = dict(row.get("metadata") or {})
        nli_prefilter = dict(metadata.get("nli_prefilter") or {})
        score = nli_prefilter.get("score")
        if score is None:
            continue
        scored_rows.append((float(score), row))

    if not scored_rows:
        return

    scored_rows.sort(key=lambda item: item[0])
    labels = _resolve_reverse_confidence_labels(len(scored_rows))
    for label, (_, row) in zip(labels, scored_rows):
        parsed_output = row.get("parsed_output")
        if not parsed_output:
            continue
        updated_output = validate_structured_payload(parsed_output).model_copy(
            update={"confidence": ConfidenceLevel(label)}
        )
        canonical_output = to_canonical_json(updated_output, spec=spec)
        completion_tokens = count_text_tokens(canonical_output, validation_cfg.tokenizer_name)
        completion_over_budget = completion_tokens > validation_cfg.max_completion_tokens
        row["parsed_output"] = updated_output.model_dump(mode="json")
        row["canonical_output"] = canonical_output
        validation_report = dict(row.get("validation_report") or {})
        validation_report["derived_confidence"] = label
        validation_report["completion_tokens"] = completion_tokens
        validation_report["completion_over_budget"] = completion_over_budget
        row["validation_report"] = validation_report


def validate_answerable_candidates(
    rows: Sequence[Dict[str, Any]],
    *,
    validation_cfg: ValidationConfig,
    correctness_cfg: CorrectnessConfig,
    spec: ProtocolSpec,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Validate answerable teacher candidates and select the best one per example."""

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
    return annotated_rows, selected_rows, metrics


def validate_unanswerable_candidates(
    rows: Sequence[Dict[str, Any]],
    *,
    validation_cfg: ValidationConfig,
    spec: ProtocolSpec,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Validate unanswerable teacher candidates using the simplified refusal path."""

    annotated_rows: List[Dict[str, Any]] = []
    num_parse_ok = 0
    num_hard_pass = 0
    num_answerability_match = 0
    num_completion_over_budget = 0

    for row in rows:
        parse_result = parse_structured_output(str(row.get("raw_output") or ""), spec=spec)
        protocol_report = ProtocolCheckReport(ok=False, issues=["Parsing failed."])
        evidence_report = EvidenceCheckReport(ok=True, issues=[])
        answerability_match = None
        hard_fail_reasons: List[str] = list(parse_result.errors)
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

            gold_answerability = str(row.get("answerability_label") or "").strip()
            if gold_answerability:
                answerability_match = parse_result.parsed.answerability.value == gold_answerability
                num_answerability_match += int(bool(answerability_match))

            if not protocol_report.ok:
                hard_fail_reasons.extend(protocol_report.issues)
            if answerability_match is False:
                hard_fail_reasons.append("Predicted answerability does not match the target label.")
            if completion_over_budget:
                num_completion_over_budget += 1
                hard_fail_reasons.append(
                    f"Canonical completion exceeds max_completion_tokens={validation_cfg.max_completion_tokens}."
                )

        hard_pass = len(hard_fail_reasons) == 0
        num_hard_pass += int(hard_pass)
        report = ValidationReport(
            parse_ok=parse_result.ok,
            overall_ok=hard_pass,
            hard_pass=hard_pass,
            protocol=protocol_report,
            evidence=evidence_report,
            correctness=None,
            semantics=None,
            answerability_match=answerability_match,
            derived_confidence=None,
            selection_score=0.0,
            hard_fail_reasons=hard_fail_reasons,
            soft_metrics={},
            semantic_margin=None,
            semantic_decision_source=None,
            semantic_filter_applied=False,
            completion_tokens=completion_tokens,
            completion_over_budget=completion_over_budget,
            issues=list(hard_fail_reasons),
        )
        annotated_rows.append(
            {
                **row,
                "parse_ok": bool(parse_result.ok),
                "parsed_output": parsed_payload,
                "canonical_output": canonical_output,
                "validation_report": asdict(report),
            }
        )

    selected_rows = select_first_valid_unanswerable_candidate(annotated_rows)
    _refresh_selected_unanswerable_confidence(rows=selected_rows, validation_cfg=validation_cfg, spec=spec)
    metrics = {
        "num_rows": len(rows),
        "num_parse_ok": num_parse_ok,
        "num_hard_pass": num_hard_pass,
        "num_answerability_match": num_answerability_match,
        "num_completion_over_budget": num_completion_over_budget,
        "num_selected": len(selected_rows),
    }
    return annotated_rows, selected_rows, metrics


def validate_mixed_teacher_candidates(
    *,
    in_path: str,
    out_path: str,
    selected_out_path: Optional[str],
    metrics_out: Optional[str],
    validation_cfg: ValidationConfig,
    correctness_cfg: CorrectnessConfig = CorrectnessConfig(),
    spec: ProtocolSpec = DEFAULT_PROTOCOL_SPEC,
) -> Dict[str, Any]:
    """Validate mixed teacher candidates by dispatching on answerability_label."""

    if validation_cfg.semantic_decision_source not in {"full_binary", "reject_aware"}:
        raise ValueError(f"Unsupported semantic_decision_source: {validation_cfg.semantic_decision_source}")

    rows = [dict(row) for row in read_jsonl(in_path)]
    indexed_rows = [{"__input_order": idx, **row} for idx, row in enumerate(rows)]

    answerable_rows = [row for row in indexed_rows if str(row.get("answerability_label") or "") == "answerable"]
    unanswerable_rows = [row for row in indexed_rows if str(row.get("answerability_label") or "") == "unanswerable"]
    unsupported_labels = [
        str(row.get("answerability_label") or "")
        for row in indexed_rows
        if str(row.get("answerability_label") or "") not in {"answerable", "unanswerable"}
    ]
    if unsupported_labels:
        raise ValueError(f"Unsupported answerability labels encountered: {sorted(set(unsupported_labels))}")

    answerable_annotated, answerable_selected, answerable_metrics = validate_answerable_candidates(
        answerable_rows,
        validation_cfg=validation_cfg,
        correctness_cfg=correctness_cfg,
        spec=spec,
    )
    unanswerable_annotated, unanswerable_selected, unanswerable_metrics = validate_unanswerable_candidates(
        unanswerable_rows,
        validation_cfg=validation_cfg,
        spec=spec,
    )

    merged_annotated = sorted(answerable_annotated + unanswerable_annotated, key=lambda row: int(row["__input_order"]))
    for row in merged_annotated:
        row.pop("__input_order", None)
    selected_rows = answerable_selected + unanswerable_selected
    for row in selected_rows:
        row.pop("__input_order", None)

    metrics = {
        "num_rows": len(rows),
        "num_selected": len(selected_rows),
        "answerable": answerable_metrics,
        "unanswerable": unanswerable_metrics,
    }
    write_jsonl(out_path, merged_annotated)
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
