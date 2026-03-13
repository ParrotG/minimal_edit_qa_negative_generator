from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from dataio import write_json, write_jsonl
from project_config import PROJECT_SETTINGS
from project_config.resolve import resolve_eval_args, resolve_matcher_args, resolve_nli_args
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
from .summary_builders import (
    build_test_summary_rows as _build_test_summary_rows_common,
    build_validation_summary_rows as _build_validation_summary_rows_common,
    constraint_pass as _constraint_pass_common,
    select_best_checkpoint as _select_best_checkpoint_common,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_data_path", type=str, required=True, help="Validation dataset path.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Test dataset path.")
    parser.add_argument("--validation_split", type=str, default=None)
    parser.add_argument("--test_split", type=str, default=None)
    parser.add_argument("--validation_max_samples", type=int, default=None)
    parser.add_argument("--test_max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--lora_ckpt_path", type=str, default=None)
    parser.add_argument("--lora_ckpt_list_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--structured_batch_size", type=int, default=None)
    parser.add_argument("--task_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--structured_max_new_tokens", type=int, default=None)
    parser.add_argument("--structured_max_attempts", type=int, default=None)
    parser.add_argument("--base_protocol_max_new_tokens", type=int, default=None)
    parser.add_argument("--base_task_max_new_tokens", type=int, default=None)
    parser.add_argument("--base_task_think_max_new_tokens", type=int, default=None)
    parser.add_argument("--record_token_usage", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fewshot_k", type=int, default=None)
    parser.add_argument("--protocol_max_attempts", type=int, default=None)
    parser.add_argument("--protocol_temperature", type=float, default=None)
    parser.add_argument("--structured_temperature", type=float, default=None)
    parser.add_argument("--task_temperature", type=float, default=None)

    parser.add_argument("--parse_ok_threshold", type=float, default=None)
    parser.add_argument("--protocol_ok_threshold", type=float, default=None)
    parser.add_argument("--evidence_ok_threshold", type=float, default=None)

    parser.add_argument("--enable_semantics", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--semantic_decision_source", type=str, default=None, choices=["full_binary", "reject_aware"])
    parser.add_argument("--semantic_match_f1_threshold", type=float, default=None)
    parser.add_argument("--nli_model_name", type=str, default=None)
    parser.add_argument("--nli_device", type=str, default=None)
    parser.add_argument("--nli_batch_size", type=int, default=None)
    parser.add_argument("--nli_max_length", type=int, default=None)
    parser.add_argument("--nli_fp16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--full_margin_threshold", type=float, default=None)
    parser.add_argument("--reject_margin_threshold", type=float, default=None)
    parser.add_argument("--reject_band_half_width", type=float, default=None)
    parser.add_argument("--qa_fail_as_negative", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--qa_check_answer_type", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--qa_spacy_model", type=str, default=None)
    parser.add_argument("--matcher_model_name", type=str, default=None)
    parser.add_argument("--matcher_threshold", type=float, default=None)

    parser.add_argument("--api_model_name", type=str, default=None)
    parser.add_argument("--api_base_url", type=str, default=None)
    parser.add_argument("--api_key_env", type=str, default=None)
    parser.add_argument("--api_timeout_seconds", type=float, default=None)
    parser.add_argument("--api_max_concurrency", type=int, default=None)
    parser.add_argument("--api_max_retries", type=int, default=None)
    parser.add_argument("--api_backoff_base_seconds", type=float, default=None)
    parser.add_argument("--api_backoff_max_seconds", type=float, default=None)
    parser.add_argument("--api_extraction_max_new_tokens", type=int, default=None)
    parser.add_argument("--api_temperature", type=float, default=None)
    parser.add_argument("--api_top_p", type=float, default=None)
    parser.add_argument("--api_seed", type=int, default=None)
    parser.add_argument("--error_log_dir", type=str, default=None)
    parser.add_argument("--extraction_max_attempts", type=int, default=None)

    parser.add_argument("--deepeval_judge_model", type=str, default=None)
    parser.add_argument("--deepeval_threshold", type=float, default=None)
    parser.add_argument("--deepeval_max_concurrent", type=int, default=None)
    parser.add_argument("--deepeval_throttle_value", type=float, default=None)
    args = parser.parse_args()
    args = resolve_nli_args(args)
    args = resolve_matcher_args(args)
    args = resolve_eval_args(args, preset="selection")
    if args.seed is None:
        args.seed = PROJECT_SETTINGS.generation.seed
    if args.base_model is None:
        args.base_model = PROJECT_SETTINGS.model.target_training_llm
    if args.validation_split is None:
        args.validation_split = PROJECT_SETTINGS.eval.structured_generation.split
    if args.test_split is None:
        args.test_split = PROJECT_SETTINGS.eval.flat_generation.split
    if args.validation_max_samples is None:
        args.validation_max_samples = PROJECT_SETTINGS.eval.structured_generation.max_samples
    if args.test_max_samples is None:
        args.test_max_samples = PROJECT_SETTINGS.eval.flat_generation.max_samples
    if args.structured_batch_size is None:
        args.structured_batch_size = PROJECT_SETTINGS.eval.structured_generation.batch_size
    if args.task_batch_size is None:
        args.task_batch_size = PROJECT_SETTINGS.eval.flat_generation.batch_size
    if args.max_length is None:
        args.max_length = PROJECT_SETTINGS.training.max_length
    if args.structured_max_new_tokens is None:
        args.structured_max_new_tokens = PROJECT_SETTINGS.eval.structured_generation.max_new_tokens
    if args.structured_max_attempts is None:
        args.structured_max_attempts = PROJECT_SETTINGS.eval.structured_generation.max_attempts
    if args.base_protocol_max_new_tokens is None:
        args.base_protocol_max_new_tokens = PROJECT_SETTINGS.eval.base_protocol.max_new_tokens
    if args.base_task_max_new_tokens is None:
        args.base_task_max_new_tokens = PROJECT_SETTINGS.eval.flat_generation.max_new_tokens
    if args.base_task_think_max_new_tokens is None:
        args.base_task_think_max_new_tokens = PROJECT_SETTINGS.eval.flat_generation.think_max_new_tokens
    if args.record_token_usage is None:
        args.record_token_usage = PROJECT_SETTINGS.token_budget.record_token_usage
    if args.fewshot_k is None:
        args.fewshot_k = PROJECT_SETTINGS.eval.base_protocol.fewshot_k
    if args.protocol_max_attempts is None:
        args.protocol_max_attempts = PROJECT_SETTINGS.eval.base_protocol.max_attempts
    if args.protocol_temperature is None:
        args.protocol_temperature = PROJECT_SETTINGS.eval.base_protocol.temperature
    if args.structured_temperature is None:
        args.structured_temperature = PROJECT_SETTINGS.eval.structured_generation.temperature
    if args.task_temperature is None:
        args.task_temperature = PROJECT_SETTINGS.eval.flat_generation.temperature
    if args.enable_semantics is None:
        args.enable_semantics = PROJECT_SETTINGS.eval.grounded.enable_semantics
    if args.semantic_decision_source is None:
        args.semantic_decision_source = PROJECT_SETTINGS.eval.grounded.semantic_decision_source
    if args.semantic_match_f1_threshold is None:
        args.semantic_match_f1_threshold = PROJECT_SETTINGS.correctness.semantic_match_f1_threshold
    if args.api_model_name is None:
        args.api_model_name = PROJECT_SETTINGS.answer_extraction_api.model_name
    if args.api_base_url is None:
        args.api_base_url = PROJECT_SETTINGS.answer_extraction_api.base_url
    if args.api_key_env is None:
        args.api_key_env = PROJECT_SETTINGS.answer_extraction_api.api_key_env
    if args.api_timeout_seconds is None:
        args.api_timeout_seconds = PROJECT_SETTINGS.answer_extraction_api.timeout_seconds
    if args.api_max_concurrency is None:
        args.api_max_concurrency = PROJECT_SETTINGS.answer_extraction_api.max_concurrency
    if args.api_max_retries is None:
        args.api_max_retries = PROJECT_SETTINGS.answer_extraction_api.max_retries
    if args.api_backoff_base_seconds is None:
        args.api_backoff_base_seconds = PROJECT_SETTINGS.answer_extraction_api.backoff_base_seconds
    if args.api_backoff_max_seconds is None:
        args.api_backoff_max_seconds = PROJECT_SETTINGS.answer_extraction_api.backoff_max_seconds
    if args.api_extraction_max_new_tokens is None:
        args.api_extraction_max_new_tokens = PROJECT_SETTINGS.answer_extraction_api.max_tokens
    if args.api_temperature is None:
        args.api_temperature = PROJECT_SETTINGS.answer_extraction_api.temperature
    if args.api_top_p is None:
        args.api_top_p = PROJECT_SETTINGS.answer_extraction_api.top_p
    if args.api_seed is None:
        args.api_seed = PROJECT_SETTINGS.answer_extraction_api.seed
    if args.error_log_dir is None:
        args.error_log_dir = PROJECT_SETTINGS.paths.error_log_dir
    if args.extraction_max_attempts is None:
        args.extraction_max_attempts = PROJECT_SETTINGS.answer_extraction_api.max_attempts
    if args.deepeval_judge_model is None:
        args.deepeval_judge_model = PROJECT_SETTINGS.eval.deepeval.judge_model
    if args.deepeval_threshold is None:
        args.deepeval_threshold = PROJECT_SETTINGS.eval.deepeval.threshold
    if args.deepeval_max_concurrent is None:
        args.deepeval_max_concurrent = PROJECT_SETTINGS.eval.deepeval.max_concurrent
    if args.deepeval_throttle_value is None:
        args.deepeval_throttle_value = PROJECT_SETTINGS.eval.deepeval.throttle_value
    return args


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _model_key(row: Dict[str, Any]) -> Tuple[str, int, str, str, str]:
    return (
        str(row.get("model_tag") or ""),
        int(row.get("model_step") or 0),
        str(row.get("model_path") or ""),
        str(row.get("eval_track") or ""),
        str(row.get("eval_variant") or ""),
    )


def _is_ckpt_row(row: Dict[str, Any]) -> bool:
    return str(row.get("model_tag") or "").strip() != "base"


def _select_best_checkpoint(
    *,
    eval_rows: Sequence[Dict[str, Any]],
    loss_rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return _select_best_checkpoint_common(
        eval_rows=eval_rows,
        loss_rows=loss_rows,
        parse_ok_threshold=float(args.parse_ok_threshold),
        protocol_ok_threshold=float(args.protocol_ok_threshold),
        evidence_ok_threshold=float(args.evidence_ok_threshold),
    )


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


def _constraint_pass(row: Dict[str, Any], args: argparse.Namespace) -> bool:
    return _constraint_pass_common(
        row,
        parse_ok_threshold=float(args.parse_ok_threshold),
        protocol_ok_threshold=float(args.protocol_ok_threshold),
        evidence_ok_threshold=float(args.evidence_ok_threshold),
    )


def _build_validation_summary_rows(
    *,
    eval_rows: Sequence[Dict[str, Any]],
    loss_rows: Sequence[Dict[str, Any]],
    selection: Dict[str, Any],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    return _build_validation_summary_rows_common(
        eval_rows=eval_rows,
        loss_rows=loss_rows,
        selection=selection,
        parse_ok_threshold=float(args.parse_ok_threshold),
        protocol_ok_threshold=float(args.protocol_ok_threshold),
        evidence_ok_threshold=float(args.evidence_ok_threshold),
    )


def _build_test_summary_rows(
    *,
    track_rows: Sequence[Dict[str, Any]],
    detail_rows: Sequence[Dict[str, Any]],
    deepeval_detail_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return _build_test_summary_rows_common(
        track_rows=track_rows,
        detail_rows=detail_rows,
        deepeval_detail_rows=deepeval_detail_rows,
    )


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
        matcher_threshold=args.matcher_threshold,
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
        matcher_threshold=args.matcher_threshold,
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
        retry_on_protocol_fail=(prompt_mode != "infer") or int(args.structured_max_attempts) > 1,
        max_attempts=args.protocol_max_attempts if prompt_mode != "infer" else int(args.structured_max_attempts),
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
        include_base=False,
        batch_size=args.structured_batch_size,
        max_length=args.max_length,
        eval_track="sft_structured",
        eval_variant="checkpoint",
        out_csv="",
        details_out=None,
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
            include_base=False,
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
    validation_summary_path = report_dir / "validation_summary.csv"
    validation_summary_rows = _build_validation_summary_rows(
        eval_rows=sft_val_curve,
        loss_rows=loss_rows,
        selection=selection,
        args=args,
    )
    write_csv(validation_summary_rows, str(validation_summary_path))

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
    test_summary_source_rows: List[Dict[str, Any]] = list(best_test_curve)
    test_summary_detail_rows: List[Dict[str, Any]] = list(best_test_details)
    test_summary_deepeval_detail_rows: List[Dict[str, Any]] = list(best_test_deepeval_details)

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
    test_summary_source_rows.extend(base_protocol_test_curve)
    test_summary_detail_rows.extend(base_protocol_test_details)
    test_summary_deepeval_detail_rows.extend(base_protocol_test_deepeval_details)

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
        test_summary_source_rows.extend(task_curve)
        test_summary_detail_rows.extend(task_details)
        test_summary_deepeval_detail_rows.extend(task_deepeval_details)

    test_summary_path = report_dir / "test_summary.csv"
    test_summary_rows = _build_test_summary_rows(
        track_rows=test_summary_source_rows,
        detail_rows=test_summary_detail_rows,
        deepeval_detail_rows=test_summary_deepeval_detail_rows,
    )
    write_csv(test_summary_rows, str(test_summary_path))

    manifest = {
        "validation": {
            "loss_curve": str(loss_curve_path),
            "structured_curve": str(sft_val_curve_path),
            "summary_csv": str(validation_summary_path),
        },
        "selection": selection,
        "report": {
            "validation_summary_csv": str(validation_summary_path),
            "test_summary_csv": str(test_summary_path),
        },
        "test_artifacts": test_artifacts,
    }
    write_json(str(report_dir / "manifest.json"), manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = run_sft_eval_report(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
