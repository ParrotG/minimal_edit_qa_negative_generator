from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from dataio import write_json, write_jsonl
from project_config import PROJECT_SETTINGS
from qa_checks.correctness import CorrectnessConfig
from qa_judge.config import JudgeConfig, NLIConfig

from .answer_extraction import AnswerExtractionConfig
from .common import write_csv
from .correctness_transformer_matcher import TransformerMatcherConfig
from .eval_deepeval_hallucination import run_deepeval_hallucination
from .eval_grounded_qa import run_grounded_qa_evaluation
from .eval_sft_loss_curve import run_sft_loss_curve
from .eval_task_content_baseline import run_task_content_baseline
from .generate_answers import run_answer_generation
from .generate_structured_answers import run_structured_generation
from .merge_eval_curves import merge_curve_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_data_path", type=str, required=True, help="Validation dataset path.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Test dataset path.")
    parser.add_argument("--validation_split", type=str, default="validation")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--validation_max_samples", type=int, default=-1)
    parser.add_argument("--test_max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_model", type=str, default=PROJECT_SETTINGS.model.target_training_llm)
    parser.add_argument("--lora_ckpt_path", type=str, default=None)
    parser.add_argument("--lora_ckpt_list_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--structured_batch_size", type=int, default=4)
    parser.add_argument("--task_batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--structured_max_new_tokens", type=int, default=512)
    parser.add_argument("--base_protocol_max_new_tokens", type=int, default=512)
    parser.add_argument("--base_task_max_new_tokens", type=int, default=512)
    parser.add_argument("--base_task_think_max_new_tokens", type=int, default=1024)
    parser.add_argument("--record_token_usage", action=argparse.BooleanOptionalAction, default=PROJECT_SETTINGS.token_budget.record_token_usage)
    parser.add_argument("--fewshot_k", type=int, default=2)
    parser.add_argument("--protocol_max_attempts", type=int, default=3)
    parser.add_argument("--protocol_temperature", type=float, default=0.2)
    parser.add_argument("--structured_temperature", type=float, default=0.0)
    parser.add_argument("--task_temperature", type=float, default=0.0)

    parser.add_argument("--parse_ok_threshold", type=float, default=0.95)
    parser.add_argument("--protocol_ok_threshold", type=float, default=0.98)
    parser.add_argument("--evidence_ok_threshold", type=float, default=0.95)

    parser.add_argument("--enable_semantics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--semantic_decision_source", type=str, default="full_binary", choices=["full_binary", "reject_aware"])
    parser.add_argument("--semantic_match_f1_threshold", type=float, default=CorrectnessConfig.semantic_match_f1_threshold)
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
    parser.add_argument("--api_extraction_max_new_tokens", type=int, default=AnswerExtractionConfig.max_new_tokens)
    parser.add_argument("--api_temperature", type=float, default=AnswerExtractionConfig.temperature)
    parser.add_argument("--api_top_p", type=float, default=AnswerExtractionConfig.top_p)
    parser.add_argument("--api_seed", type=int, default=AnswerExtractionConfig.seed)
    parser.add_argument("--error_log_dir", type=str, default=AnswerExtractionConfig.error_log_dir)
    parser.add_argument("--extraction_max_attempts", type=int, default=AnswerExtractionConfig.max_attempts)

    parser.add_argument("--deepeval_judge_model", type=str, default="gpt-5.2")
    parser.add_argument("--deepeval_threshold", type=float, default=0.5)
    parser.add_argument("--deepeval_max_concurrent", type=int, default=4)
    parser.add_argument("--deepeval_throttle_value", type=float, default=3.0)
    return parser.parse_args()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _model_key(row: Dict[str, Any]) -> Tuple[str, int, str, str, str]:
    return (
        str(row.get("model_tag") or ""),
        int(row.get("model_step") or 0),
        str(row.get("model_path") or ""),
        str(row.get("eval_track") or ""),
        str(row.get("eval_variant") or ""),
    )


def _safe_metric(value: Any, *, nan_for_max: float = float("-inf"), nan_for_min: float = float("inf")) -> Tuple[float, float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return nan_for_max, nan_for_min
    if math.isnan(numeric):
        return nan_for_max, nan_for_min
    return numeric, numeric


def _is_ckpt_row(row: Dict[str, Any]) -> bool:
    return str(row.get("model_tag") or "").strip() != "base"


def _select_best_checkpoint(
    *,
    eval_rows: Sequence[Dict[str, Any]],
    loss_rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    loss_by_key = {_model_key(row): dict(row) for row in loss_rows}
    candidate_rows = [dict(row) for row in eval_rows if _is_ckpt_row(row)]
    if not candidate_rows:
        raise RuntimeError("No checkpoint rows were found in the validation structured evaluation curve.")

    def constraint_pass(row: Dict[str, Any]) -> bool:
        parse_ok = float(row.get("parse_ok_rate") or float("nan"))
        protocol_ok = float(row.get("protocol_ok_rate_given_parse_ok") or float("nan"))
        evidence_ok = float(row.get("evidence_substring_ok_rate") or float("nan"))
        return (
            not math.isnan(parse_ok)
            and not math.isnan(protocol_ok)
            and not math.isnan(evidence_ok)
            and parse_ok >= float(args.parse_ok_threshold)
            and protocol_ok >= float(args.protocol_ok_threshold)
            and evidence_ok >= float(args.evidence_ok_threshold)
        )

    passed = [row for row in candidate_rows if constraint_pass(row)]
    active = passed if passed else candidate_rows

    def sort_key(row: Dict[str, Any]) -> Tuple[float, float, float, float, int]:
        loss_row = loss_by_key.get(_model_key(row), {})
        correctness, _ = _safe_metric(row.get("correctness_reviewed_rate"))
        answerability, _ = _safe_metric(row.get("answerability_accuracy"))
        semantic, _ = _safe_metric(row.get("semantic_yes_rate"))
        _, loss_value = _safe_metric(loss_row.get("mean_loss"))
        return (
            correctness,
            answerability,
            semantic,
            -loss_value,
            -int(row.get("model_step") or 0),
        )

    selected = max(active, key=sort_key)
    selection = {
        "constraint_satisfied": bool(passed),
        "selected_model_tag": selected["model_tag"],
        "selected_model_step": int(selected["model_step"]),
        "selected_model_path": selected["model_path"],
        "selected_eval_track": selected.get("eval_track"),
        "selected_eval_variant": selected.get("eval_variant"),
        "selection_metrics": {
            "correctness_reviewed_rate": selected.get("correctness_reviewed_rate"),
            "answerability_accuracy": selected.get("answerability_accuracy"),
            "semantic_yes_rate": selected.get("semantic_yes_rate"),
            "mean_loss": (loss_by_key.get(_model_key(selected), {}) or {}).get("mean_loss"),
        },
        "num_candidates": len(candidate_rows),
        "num_constraint_pass": len(passed),
        "candidate_rows": candidate_rows,
    }
    return selection


def _write_csv_and_jsonl(
    *,
    csv_path: Path | None = None,
    csv_rows: Sequence[Dict[str, Any]] | None = None,
    jsonl_path: Path | None = None,
    jsonl_rows: Sequence[Dict[str, Any]] | None = None,
) -> None:
    if csv_path is not None and csv_rows is not None:
        write_csv(list(csv_rows), str(csv_path))
    if jsonl_path is not None and jsonl_rows is not None:
        write_jsonl(str(jsonl_path), list(jsonl_rows))


def _grounded_eval_args(
    *,
    generated_path: str,
    split: str,
    max_samples: int,
    seed: int,
    metrics_out: str,
    details_out: str | None,
    confidence_out: str | None,
    args: argparse.Namespace,
) -> SimpleNamespace:
    return SimpleNamespace(
        generated_path=generated_path,
        split=split,
        max_samples=max_samples,
        seed=seed,
        enable_semantics=bool(args.enable_semantics),
        semantic_decision_source=args.semantic_decision_source,
        semantic_match_f1_threshold=args.semantic_match_f1_threshold,
        nli_model_name=args.nli_model_name,
        nli_device=args.nli_device,
        nli_batch_size=args.nli_batch_size,
        nli_max_length=args.nli_max_length,
        nli_fp16=bool(args.nli_fp16),
        temperature=args.temperature,
        full_margin_threshold=args.full_margin_threshold,
        reject_margin_threshold=args.reject_margin_threshold,
        reject_band_half_width=args.reject_band_half_width,
        qa_fail_as_negative=bool(args.qa_fail_as_negative),
        qa_check_answer_type=bool(args.qa_check_answer_type),
        qa_spacy_model=args.qa_spacy_model,
        matcher_model_name=args.matcher_model_name,
        metrics_out=metrics_out,
        details_out=details_out,
        confidence_out=confidence_out,
    )


def _task_eval_args(
    *,
    generated_path: str,
    split: str,
    max_samples: int,
    seed: int,
    metrics_out: str,
    details_out: str | None,
    args: argparse.Namespace,
) -> SimpleNamespace:
    return SimpleNamespace(
        generated_path=generated_path,
        split=split,
        max_samples=max_samples,
        seed=seed,
        semantic_decision_source=args.semantic_decision_source,
        semantic_match_f1_threshold=args.semantic_match_f1_threshold,
        nli_model_name=args.nli_model_name,
        nli_device=args.nli_device,
        nli_batch_size=args.nli_batch_size,
        nli_max_length=args.nli_max_length,
        nli_fp16=bool(args.nli_fp16),
        temperature=args.temperature,
        full_margin_threshold=args.full_margin_threshold,
        reject_margin_threshold=args.reject_margin_threshold,
        reject_band_half_width=args.reject_band_half_width,
        qa_fail_as_negative=bool(args.qa_fail_as_negative),
        qa_check_answer_type=bool(args.qa_check_answer_type),
        qa_spacy_model=args.qa_spacy_model,
        matcher_model_name=args.matcher_model_name,
        api_model_name=args.api_model_name,
        api_base_url=args.api_base_url,
        api_key_env=args.api_key_env,
        api_timeout_seconds=args.api_timeout_seconds,
        api_max_concurrency=args.api_max_concurrency,
        api_max_retries=args.api_max_retries,
        api_backoff_base_seconds=args.api_backoff_base_seconds,
        api_backoff_max_seconds=args.api_backoff_max_seconds,
        api_max_new_tokens=args.api_extraction_max_new_tokens,
        api_temperature=args.api_temperature,
        api_top_p=args.api_top_p,
        api_seed=args.api_seed,
        error_log_dir=args.error_log_dir,
        extraction_max_attempts=args.extraction_max_attempts,
        metrics_out=metrics_out,
        details_out=details_out,
    )


def _deepeval_args(
    *,
    generated_path: str,
    split: str,
    max_samples: int,
    seed: int,
    input_mode: str,
    answer_source: str,
    metrics_out: str,
    details_out: str | None,
    args: argparse.Namespace,
) -> SimpleNamespace:
    return SimpleNamespace(
        generated_path=generated_path,
        split=split,
        max_samples=max_samples,
        seed=seed,
        input_mode=input_mode,
        answer_source=answer_source,
        judge_model=args.deepeval_judge_model,
        threshold=args.deepeval_threshold,
        max_concurrent=args.deepeval_max_concurrent,
        throttle_value=args.deepeval_throttle_value,
        metrics_out=metrics_out,
        details_out=details_out,
    )


def _structured_generation_args(
    *,
    data_path: str,
    split: str,
    max_samples: int,
    base_model: str,
    lora_ckpt_path: str | None,
    lora_ckpt_list_path: str | None,
    include_base: bool,
    max_new_tokens: int,
    prompt_mode: str,
    eval_track: str,
    eval_variant: str,
    args: argparse.Namespace,
) -> SimpleNamespace:
    return SimpleNamespace(
        data_path=data_path,
        split=split,
        max_samples=max_samples,
        seed=args.seed,
        base_model=base_model,
        lora_ckpt_path=lora_ckpt_path,
        lora_ckpt_list_path=lora_ckpt_list_path,
        include_base=include_base,
        batch_size=args.structured_batch_size,
        max_new_tokens=max_new_tokens,
        temperature=args.protocol_temperature if prompt_mode != "infer" else args.structured_temperature,
        top_p=1.0 if prompt_mode == "infer" else 0.95,
        top_k=None,
        min_p=None,
        repetition_penalty=1.0,
        prompt_mode=prompt_mode,
        fewshot_k=args.fewshot_k,
        retry_on_protocol_fail=(prompt_mode != "infer"),
        max_attempts=args.protocol_max_attempts if prompt_mode != "infer" else 1,
        use_chat_template=True,
        enable_thinking=False,
        strip_think_tags=True,
        strip_role_markers=True,
        record_token_usage=bool(args.record_token_usage),
        eval_track=eval_track,
        eval_variant=eval_variant,
        out_jsonl="",
    )


def _task_generation_args(
    *,
    data_path: str,
    split: str,
    max_samples: int,
    max_new_tokens: int,
    enable_thinking: bool,
    eval_variant: str,
    args: argparse.Namespace,
) -> SimpleNamespace:
    return SimpleNamespace(
        data_path=data_path,
        split=split,
        max_samples=max_samples,
        seed=args.seed,
        base_model=args.base_model,
        lora_ckpt_path=None,
        lora_ckpt_list_path=None,
        include_base=True,
        batch_size=args.task_batch_size,
        max_new_tokens=max_new_tokens,
        temperature=args.task_temperature,
        top_p=1.0,
        top_k=None,
        min_p=None,
        repetition_penalty=1.0,
        use_chat_template=True,
        enable_thinking=enable_thinking,
        strip_think_tags=True,
        strip_role_markers=True,
        encourage_refusal=True,
        record_token_usage=bool(args.record_token_usage),
        eval_track="base_task",
        eval_variant=eval_variant,
        out_jsonl="",
    )


def _loss_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        data_path=args.validation_data_path,
        split=args.validation_split,
        max_samples=args.validation_max_samples,
        seed=args.seed,
        base_model=args.base_model,
        lora_ckpt_path=args.lora_ckpt_path,
        lora_ckpt_list_path=args.lora_ckpt_list_path,
        include_base=True,
        batch_size=args.structured_batch_size,
        max_length=args.max_length,
        eval_track="sft_structured",
        eval_variant="checkpoint",
        out_csv="",
        details_out=None,
    )


def _build_report_markdown(
    *,
    selection: Dict[str, Any],
    validation_curve_path: str,
    merged_curve_path: str,
    test_artifacts: Dict[str, str],
) -> str:
    selected = selection["selected_model_path"]
    metrics = selection["selection_metrics"]
    return "\n".join(
        [
            "# SFT Evaluation Report",
            "",
            "## Selected Checkpoint",
            f"- Path: `{selected}`",
            f"- Constraint satisfied: `{selection['constraint_satisfied']}`",
            f"- correctness_reviewed_rate: `{metrics.get('correctness_reviewed_rate')}`",
            f"- answerability_accuracy: `{metrics.get('answerability_accuracy')}`",
            f"- semantic_yes_rate: `{metrics.get('semantic_yes_rate')}`",
            f"- mean_loss: `{metrics.get('mean_loss')}`",
            "",
            "## Validation Artifacts",
            f"- Structured curve: `{validation_curve_path}`",
            f"- Merged comparison curve: `{merged_curve_path}`",
            "",
            "## Test Artifacts",
            *[f"- {name}: `{path}`" for name, path in sorted(test_artifacts.items())],
            "",
        ]
    )


def run_sft_eval_report(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the end-to-end SFT evaluation/report workflow."""

    out_dir = Path(args.out_dir)
    validation_dir = out_dir / "validation"
    test_dir = out_dir / "test"
    report_dir = out_dir / "report"
    for path in (validation_dir, test_dir, report_dir):
        _ensure_dir(path)

    loss_rows, loss_details = run_sft_loss_curve(_loss_args(args))
    loss_curve_path = validation_dir / "sft_val_loss_curve.csv"
    loss_details_path = validation_dir / "sft_val_loss_details.jsonl"
    _write_csv_and_jsonl(csv_path=loss_curve_path, csv_rows=loss_rows, jsonl_path=loss_details_path, jsonl_rows=loss_details)

    sft_val_generations = run_structured_generation(
        _structured_generation_args(
            data_path=args.validation_data_path,
            split=args.validation_split,
            max_samples=args.validation_max_samples,
            base_model=args.base_model,
            lora_ckpt_path=args.lora_ckpt_path,
            lora_ckpt_list_path=args.lora_ckpt_list_path,
            include_base=True,
            max_new_tokens=args.structured_max_new_tokens,
            prompt_mode="infer",
            eval_track="sft_structured",
            eval_variant="checkpoint",
            args=args,
        )
    )
    sft_val_generations_path = validation_dir / "sft_val_structured_generations.jsonl"
    write_jsonl(str(sft_val_generations_path), sft_val_generations)

    sft_val_curve, sft_val_details, sft_val_confidence = run_grounded_qa_evaluation(
        _grounded_eval_args(
            generated_path=str(sft_val_generations_path),
            split="train",
            max_samples=-1,
            seed=args.seed,
            metrics_out=str(validation_dir / "sft_val_structured_curve.csv"),
            details_out=str(validation_dir / "sft_val_structured_details.jsonl"),
            confidence_out=str(validation_dir / "sft_val_confidence_analysis.json"),
            args=args,
        )
    )
    sft_val_curve_path = validation_dir / "sft_val_structured_curve.csv"
    sft_val_details_path = validation_dir / "sft_val_structured_details.jsonl"
    sft_val_confidence_path = validation_dir / "sft_val_confidence_analysis.json"
    _write_csv_and_jsonl(csv_path=sft_val_curve_path, csv_rows=sft_val_curve, jsonl_path=sft_val_details_path, jsonl_rows=sft_val_details)
    write_json(str(sft_val_confidence_path), {"models": sft_val_confidence})

    selection = _select_best_checkpoint(eval_rows=sft_val_curve, loss_rows=loss_rows, args=args)
    selection_path = report_dir / "selection.json"
    write_json(str(selection_path), selection)

    base_protocol_generations = run_structured_generation(
        _structured_generation_args(
            data_path=args.validation_data_path,
            split=args.validation_split,
            max_samples=args.validation_max_samples,
            base_model=args.base_model,
            lora_ckpt_path=None,
            lora_ckpt_list_path=None,
            include_base=True,
            max_new_tokens=args.base_protocol_max_new_tokens,
            prompt_mode="teacher_fewshot",
            eval_track="base_protocol",
            eval_variant="fewshot_retry",
            args=args,
        )
    )
    base_protocol_generation_path = validation_dir / "base_protocol_generations.jsonl"
    write_jsonl(str(base_protocol_generation_path), base_protocol_generations)
    base_protocol_curve, base_protocol_details, base_protocol_confidence = run_grounded_qa_evaluation(
        _grounded_eval_args(
            generated_path=str(base_protocol_generation_path),
            split="train",
            max_samples=-1,
            seed=args.seed,
            metrics_out=str(validation_dir / "base_protocol_curve.csv"),
            details_out=str(validation_dir / "base_protocol_details.jsonl"),
            confidence_out=str(validation_dir / "base_protocol_confidence.json"),
            args=args,
        )
    )
    base_protocol_curve_path = validation_dir / "base_protocol_curve.csv"
    _write_csv_and_jsonl(csv_path=base_protocol_curve_path, csv_rows=base_protocol_curve, jsonl_path=validation_dir / "base_protocol_details.jsonl", jsonl_rows=base_protocol_details)
    write_json(str(validation_dir / "base_protocol_confidence.json"), {"models": base_protocol_confidence})

    base_task_nothink_generations = run_answer_generation(
        _task_generation_args(
            data_path=args.validation_data_path,
            split=args.validation_split,
            max_samples=args.validation_max_samples,
            max_new_tokens=args.base_task_max_new_tokens,
            enable_thinking=False,
            eval_variant="no_think",
            args=args,
        )
    )
    base_task_nothink_generation_path = validation_dir / "base_task_nothink_generations.jsonl"
    write_jsonl(str(base_task_nothink_generation_path), base_task_nothink_generations)
    base_task_nothink_curve, base_task_nothink_details = run_task_content_baseline(
        _task_eval_args(
            generated_path=str(base_task_nothink_generation_path),
            split="train",
            max_samples=-1,
            seed=args.seed,
            metrics_out=str(validation_dir / "base_task_nothink_curve.csv"),
            details_out=str(validation_dir / "base_task_nothink_details.jsonl"),
            args=args,
        )
    )
    base_task_nothink_curve_path = validation_dir / "base_task_nothink_curve.csv"
    _write_csv_and_jsonl(csv_path=base_task_nothink_curve_path, csv_rows=base_task_nothink_curve, jsonl_path=validation_dir / "base_task_nothink_details.jsonl", jsonl_rows=base_task_nothink_details)

    base_task_think_generations = run_answer_generation(
        _task_generation_args(
            data_path=args.validation_data_path,
            split=args.validation_split,
            max_samples=args.validation_max_samples,
            max_new_tokens=args.base_task_think_max_new_tokens,
            enable_thinking=True,
            eval_variant="think",
            args=args,
        )
    )
    base_task_think_generation_path = validation_dir / "base_task_think_generations.jsonl"
    write_jsonl(str(base_task_think_generation_path), base_task_think_generations)
    base_task_think_curve, base_task_think_details = run_task_content_baseline(
        _task_eval_args(
            generated_path=str(base_task_think_generation_path),
            split="train",
            max_samples=-1,
            seed=args.seed,
            metrics_out=str(validation_dir / "base_task_think_curve.csv"),
            details_out=str(validation_dir / "base_task_think_details.jsonl"),
            args=args,
        )
    )
    base_task_think_curve_path = validation_dir / "base_task_think_curve.csv"
    _write_csv_and_jsonl(csv_path=base_task_think_curve_path, csv_rows=base_task_think_curve, jsonl_path=validation_dir / "base_task_think_details.jsonl", jsonl_rows=base_task_think_details)

    merged_curve_path = report_dir / "merged_curve.csv"
    merged_rows = merge_curve_rows(
        SimpleNamespace(
            sft_structured_csv=str(sft_val_curve_path),
            base_protocol_csv=str(base_protocol_curve_path),
            base_task_think_csv=str(base_task_think_curve_path),
            base_task_nothink_csv=str(base_task_nothink_curve_path),
            out_csv=str(merged_curve_path),
        )
    )
    write_csv(merged_rows, str(merged_curve_path))

    best_ckpt_path = str(selection["selected_model_path"])
    test_artifacts: Dict[str, str] = {}

    best_test_generations = run_structured_generation(
        _structured_generation_args(
            data_path=args.test_data_path,
            split=args.test_split,
            max_samples=args.test_max_samples,
            base_model=args.base_model,
            lora_ckpt_path=best_ckpt_path,
            lora_ckpt_list_path=None,
            include_base=False,
            max_new_tokens=args.structured_max_new_tokens,
            prompt_mode="infer",
            eval_track="sft_structured",
            eval_variant="checkpoint",
            args=args,
        )
    )
    best_test_generation_path = test_dir / "best_ckpt_structured_generations.jsonl"
    write_jsonl(str(best_test_generation_path), best_test_generations)
    best_test_curve, best_test_details, best_test_confidence = run_grounded_qa_evaluation(
        _grounded_eval_args(
            generated_path=str(best_test_generation_path),
            split="train",
            max_samples=-1,
            seed=args.seed,
            metrics_out=str(test_dir / "best_ckpt_structured_curve.csv"),
            details_out=str(test_dir / "best_ckpt_structured_details.jsonl"),
            confidence_out=str(test_dir / "best_ckpt_structured_confidence.json"),
            args=args,
        )
    )
    _write_csv_and_jsonl(csv_path=test_dir / "best_ckpt_structured_curve.csv", csv_rows=best_test_curve, jsonl_path=test_dir / "best_ckpt_structured_details.jsonl", jsonl_rows=best_test_details)
    write_json(str(test_dir / "best_ckpt_structured_confidence.json"), {"models": best_test_confidence})
    best_test_deepeval, best_test_deepeval_details = run_deepeval_hallucination(
        _deepeval_args(
            generated_path=str(best_test_generation_path),
            split="train",
            max_samples=-1,
            seed=args.seed,
            input_mode="structured",
            answer_source="rationale_plus_answer",
            metrics_out=str(test_dir / "best_ckpt_structured_deepeval_curve.csv"),
            details_out=str(test_dir / "best_ckpt_structured_deepeval_details.jsonl"),
            args=args,
        )
    )
    _write_csv_and_jsonl(csv_path=test_dir / "best_ckpt_structured_deepeval_curve.csv", csv_rows=best_test_deepeval, jsonl_path=test_dir / "best_ckpt_structured_deepeval_details.jsonl", jsonl_rows=best_test_deepeval_details)
    test_artifacts["best_ckpt_structured_curve"] = str(test_dir / "best_ckpt_structured_curve.csv")
    test_artifacts["best_ckpt_structured_deepeval"] = str(test_dir / "best_ckpt_structured_deepeval_curve.csv")

    base_protocol_test_generations = run_structured_generation(
        _structured_generation_args(
            data_path=args.test_data_path,
            split=args.test_split,
            max_samples=args.test_max_samples,
            base_model=args.base_model,
            lora_ckpt_path=None,
            lora_ckpt_list_path=None,
            include_base=True,
            max_new_tokens=args.base_protocol_max_new_tokens,
            prompt_mode="teacher_fewshot",
            eval_track="base_protocol",
            eval_variant="fewshot_retry",
            args=args,
        )
    )
    base_protocol_test_generation_path = test_dir / "base_protocol_generations.jsonl"
    write_jsonl(str(base_protocol_test_generation_path), base_protocol_test_generations)
    base_protocol_test_curve, base_protocol_test_details, base_protocol_test_confidence = run_grounded_qa_evaluation(
        _grounded_eval_args(
            generated_path=str(base_protocol_test_generation_path),
            split="train",
            max_samples=-1,
            seed=args.seed,
            metrics_out=str(test_dir / "base_protocol_curve.csv"),
            details_out=str(test_dir / "base_protocol_details.jsonl"),
            confidence_out=str(test_dir / "base_protocol_confidence.json"),
            args=args,
        )
    )
    _write_csv_and_jsonl(csv_path=test_dir / "base_protocol_curve.csv", csv_rows=base_protocol_test_curve, jsonl_path=test_dir / "base_protocol_details.jsonl", jsonl_rows=base_protocol_test_details)
    write_json(str(test_dir / "base_protocol_confidence.json"), {"models": base_protocol_test_confidence})
    base_protocol_test_deepeval, base_protocol_test_deepeval_details = run_deepeval_hallucination(
        _deepeval_args(
            generated_path=str(base_protocol_test_generation_path),
            split="train",
            max_samples=-1,
            seed=args.seed,
            input_mode="structured",
            answer_source="rationale_plus_answer",
            metrics_out=str(test_dir / "base_protocol_deepeval_curve.csv"),
            details_out=str(test_dir / "base_protocol_deepeval_details.jsonl"),
            args=args,
        )
    )
    _write_csv_and_jsonl(csv_path=test_dir / "base_protocol_deepeval_curve.csv", csv_rows=base_protocol_test_deepeval, jsonl_path=test_dir / "base_protocol_deepeval_details.jsonl", jsonl_rows=base_protocol_test_deepeval_details)
    test_artifacts["base_protocol_curve"] = str(test_dir / "base_protocol_curve.csv")
    test_artifacts["base_protocol_deepeval"] = str(test_dir / "base_protocol_deepeval_curve.csv")

    for enable_thinking, eval_variant, max_tokens in (
        (False, "no_think", args.base_task_max_new_tokens),
        (True, "think", args.base_task_think_max_new_tokens),
    ):
        generation_rows = run_answer_generation(
            _task_generation_args(
                data_path=args.test_data_path,
                split=args.test_split,
                max_samples=args.test_max_samples,
                max_new_tokens=max_tokens,
                enable_thinking=enable_thinking,
                eval_variant=eval_variant,
                args=args,
            )
        )
        generation_path = test_dir / f"base_task_{eval_variant}_generations.jsonl"
        write_jsonl(str(generation_path), generation_rows)
        task_curve, task_details = run_task_content_baseline(
            _task_eval_args(
                generated_path=str(generation_path),
                split="train",
                max_samples=-1,
                seed=args.seed,
                metrics_out=str(test_dir / f"base_task_{eval_variant}_curve.csv"),
                details_out=str(test_dir / f"base_task_{eval_variant}_details.jsonl"),
                args=args,
            )
        )
        _write_csv_and_jsonl(csv_path=test_dir / f"base_task_{eval_variant}_curve.csv", csv_rows=task_curve, jsonl_path=test_dir / f"base_task_{eval_variant}_details.jsonl", jsonl_rows=task_details)
        task_deepeval, task_deepeval_details = run_deepeval_hallucination(
            _deepeval_args(
                generated_path=str(generation_path),
                split="train",
                max_samples=-1,
                seed=args.seed,
                input_mode="flat",
                answer_source="answer",
                metrics_out=str(test_dir / f"base_task_{eval_variant}_deepeval_curve.csv"),
                details_out=str(test_dir / f"base_task_{eval_variant}_deepeval_details.jsonl"),
                args=args,
            )
        )
        _write_csv_and_jsonl(csv_path=test_dir / f"base_task_{eval_variant}_deepeval_curve.csv", csv_rows=task_deepeval, jsonl_path=test_dir / f"base_task_{eval_variant}_deepeval_details.jsonl", jsonl_rows=task_deepeval_details)
        test_artifacts[f"base_task_{eval_variant}_curve"] = str(test_dir / f"base_task_{eval_variant}_curve.csv")
        test_artifacts[f"base_task_{eval_variant}_deepeval"] = str(test_dir / f"base_task_{eval_variant}_deepeval_curve.csv")

    final_report_path = report_dir / "final_report.md"
    _write_markdown(
        final_report_path,
        _build_report_markdown(
            selection=selection,
            validation_curve_path=str(sft_val_curve_path),
            merged_curve_path=str(merged_curve_path),
            test_artifacts=test_artifacts,
        ),
    )

    manifest = {
        "validation": {
            "loss_curve": str(loss_curve_path),
            "structured_curve": str(sft_val_curve_path),
            "merged_curve": str(merged_curve_path),
        },
        "selection": selection,
        "test_artifacts": test_artifacts,
        "final_report": str(final_report_path),
    }
    write_json(str(report_dir / "manifest.json"), manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = run_sft_eval_report(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
