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
    check_reference_answer_unsupported,
    evaluate_structured_semantics,
    score_validation_report,
    select_first_valid_unanswerable_candidate,
    select_best_candidates,
)
from qa_checks.report import CorrectnessCheckReport, EvidenceCheckReport, ProtocolCheckReport, SemanticCheckReport, ValidationReport
from qa_checks.selection import assign_quantile_confidence_labels
from qa_data import (
    ConstructionConfig,
    HotpotSourceConfig,
    NegativeSamplingConfig,
    SplitConfig,
    UnanswerableBuildConfig,
    assign_split,
    build_unanswerable_examples_from_pools,
    build_answerable_example,
    derive_simple_unanswerable,
    example_from_dict,
    example_to_dict,
    iter_hotpot_rows,
    iter_split_hotpot_rows,
)
from qa_judge.config import JudgeConfig, NLIConfig
from qa_judge.judge import AnswerJudge
from qa_judge.nli import NLIVerifier
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

from .config import TeacherGenerationConfig, UnanswerablePipelineConfig, UnanswerablePrefilterConfig, ValidationConfig


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


def export_split_hotpot_rows(
    *,
    out_path: str,
    metrics_out: Optional[str],
    source_cfg: HotpotSourceConfig,
    split_cfg: SplitConfig,
) -> Dict[str, Any]:
    """Export raw Hotpot rows with stable split assignments."""

    rows = list(iter_split_hotpot_rows(source_cfg, split_cfg))
    split_counts: Dict[str, int] = {}
    for row in rows:
        split_name = str(row.get("split") or "").strip()
        split_counts[split_name] = split_counts.get(split_name, 0) + 1

    metrics: Dict[str, Any] = {
        "num_rows": len(rows),
        "split_counts": split_counts,
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


def build_unanswerable_source(
    *,
    paired_answerable_path: str,
    raw_hotpot_pool_path: str,
    out_path: str,
    metrics_out: Optional[str],
    pipeline_cfg: UnanswerablePipelineConfig,
    construct_cfg: ConstructionConfig,
) -> Dict[str, Any]:
    """Build v1 unanswerable raw examples from paired and external pools."""

    paired_examples = [example_from_dict(dict(row)) for row in read_jsonl(paired_answerable_path)]
    raw_hotpot_rows = [dict(row) for row in read_jsonl(raw_hotpot_pool_path)]
    build_cfg = UnanswerableBuildConfig(
        paired_fraction=pipeline_cfg.paired_fraction,
        max_total_examples=pipeline_cfg.max_total_examples,
        replace_supporting_facts_min=pipeline_cfg.replace_supporting_facts_min,
        replace_supporting_facts_max=pipeline_cfg.replace_supporting_facts_max,
        same_doc_candidate_radius=pipeline_cfg.same_doc_candidate_radius,
        allow_same_doc_non_adjacent=pipeline_cfg.allow_same_doc_non_adjacent,
        adjacent_doc_sentence_limit=pipeline_cfg.adjacent_doc_sentence_limit,
        include_title_prefix=pipeline_cfg.include_title_prefix,
        seed=pipeline_cfg.seed,
    )

    built_examples = build_unanswerable_examples_from_pools(
        paired_examples=paired_examples,
        raw_hotpot_rows=raw_hotpot_rows,
        target_split=pipeline_cfg.target_split,
        construct_cfg=construct_cfg,
        cfg=build_cfg,
    )
    rows = [example_to_dict(example) for example in built_examples]

    origin_counts: Dict[str, int] = {}
    for row in rows:
        origin = str(((row.get("metadata") or {}).get("origin_track")) or "")
        origin_counts[origin] = origin_counts.get(origin, 0) + 1

    metrics: Dict[str, Any] = {
        "num_paired_input": sum(1 for example in paired_examples if str(example.split or "") == pipeline_cfg.target_split),
        "num_external_input": sum(1 for row in raw_hotpot_rows if str(row.get("split") or "") == pipeline_cfg.target_split),
        "num_paired_selected": origin_counts.get("paired_answerable", 0),
        "num_external_selected": origin_counts.get("external_raw", 0),
        "num_kept": len(rows),
        "origin_counts": origin_counts,
        "target_split": pipeline_cfg.target_split,
    }
    write_jsonl(out_path, rows)
    if metrics_out:
        write_json(metrics_out, metrics)
    return metrics


def _build_unanswerable_prefilter_judge(cfg: UnanswerablePrefilterConfig) -> AnswerJudge:
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


def prefilter_unanswerable_source(
    *,
    in_path: str,
    out_path: str,
    metrics_out: Optional[str],
    cfg: UnanswerablePrefilterConfig,
) -> Dict[str, Any]:
    """Keep only raw unanswerable samples whose original answer is no longer supported."""

    rows = [dict(row) for row in read_jsonl(in_path)]
    if not cfg.enable_nli_prefilter:
        write_jsonl(out_path, rows)
        metrics = {
            "num_input": len(rows),
            "num_kept": len(rows),
            "num_dropped": 0,
            "num_full_binary_no": 0,
            "num_full_binary_yes": 0,
            "mean_margin_kept": 0.0,
        }
        if metrics_out:
            write_json(metrics_out, metrics)
        return metrics

    if cfg.judge_decision_source != "full_binary":
        raise ValueError(f"Unsupported judge_decision_source: {cfg.judge_decision_source}")

    judge = _build_unanswerable_prefilter_judge(cfg)
    kept_rows: List[Dict[str, Any]] = []
    num_no = 0
    num_yes = 0
    kept_margins: List[float] = []

    for row in rows:
        report = check_reference_answer_unsupported(
            knowledge=str(row.get("knowledge") or "").strip(),
            question=str(row.get("question") or "").strip(),
            reference_answer=str(row.get("reference_answer") or "").strip(),
            judge=judge,
        )
        metadata = dict(row.get("metadata") or {})
        metadata["nli_prefilter"] = {
            "keep": bool(report.keep),
            "full_binary_decision": report.full_binary_decision,
            "margin": report.margin,
            "issues": list(report.issues),
            "judge_payload": report.judge_payload,
        }
        row["metadata"] = metadata
        if report.full_binary_decision == "no":
            num_no += 1
        else:
            num_yes += 1
        if report.keep:
            kept_rows.append(row)
            if report.margin is not None:
                kept_margins.append(float(report.margin))

    mean_margin_kept = float(sum(kept_margins) / len(kept_margins)) if kept_margins else 0.0
    metrics = {
        "num_input": len(rows),
        "num_kept": len(kept_rows),
        "num_dropped": len(rows) - len(kept_rows),
        "num_full_binary_no": num_no,
        "num_full_binary_yes": num_yes,
        "mean_margin_kept": mean_margin_kept,
    }
    write_jsonl(out_path, kept_rows)
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


def validate_unanswerable_teacher_candidates(
    *,
    in_path: str,
    out_path: str,
    selected_out_path: Optional[str],
    metrics_out: Optional[str],
    validation_cfg: ValidationConfig,
    spec: ProtocolSpec = DEFAULT_PROTOCOL_SPEC,
) -> Dict[str, Any]:
    """Validate unanswerable teacher candidates with unanswerable-specific rules."""

    rows = list(read_jsonl(in_path))
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
    metrics = {
        "num_rows": len(rows),
        "num_parse_ok": num_parse_ok,
        "num_hard_pass": num_hard_pass,
        "num_answerability_match": num_answerability_match,
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
