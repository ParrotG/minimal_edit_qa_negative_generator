from __future__ import annotations

from argparse import Namespace
from typing import Any, Optional

from .settings import PROJECT_SETTINGS


def coalesce(value: Any, fallback: Any) -> Any:
    """Return the explicit value when provided, otherwise the fallback."""

    return fallback if value is None else value


def resolve_bool(value: Optional[bool], fallback: bool) -> bool:
    """Resolve a nullable CLI bool against a central default."""

    return bool(fallback if value is None else value)


def resolve_namespace(args: Namespace, **mapping: Any) -> Namespace:
    """Fill a namespace in place from keyword fallback values."""

    for name, fallback in mapping.items():
        setattr(args, name, coalesce(getattr(args, name, None), fallback))
    return args


def resolve_source_args(args: Namespace) -> Namespace:
    """Resolve source-pipeline CLI defaults from central settings."""

    source = PROJECT_SETTINGS.source
    token_budget = PROJECT_SETTINGS.token_budget
    model = PROJECT_SETTINGS.model
    return resolve_namespace(
        args,
        dataset_name=source.dataset_name,
        train_split=source.train_split,
        validation_split=source.validation_split,
        max_train_samples=source.max_train_samples,
        max_validation_samples=source.max_validation_samples,
        seed=source.seed,
        validation_ratio=source.validation_ratio,
        test_ratio=source.test_ratio,
        data_split_hash_salt=source.data_split_hash_salt,
        answerable_ratio=source.answerable_ratio,
        unanswerable_ratio=source.unanswerable_ratio,
        both_ratio=source.both_ratio,
        answerability_hash_salt=source.answerability_hash_salt,
        window_size=source.window_size,
        max_supporting_facts=source.max_supporting_facts,
        drop_over_max_supporting_facts=source.drop_over_max_supporting_facts,
        include_title_prefix=source.include_title_prefix,
        replace_supporting_facts_min=source.replace_supporting_facts_min,
        replace_supporting_facts_max=source.replace_supporting_facts_max,
        same_doc_candidate_radius=source.same_doc_candidate_radius,
        allow_same_doc_non_adjacent=source.allow_same_doc_non_adjacent,
        adjacent_doc_sentence_limit=source.adjacent_doc_sentence_limit,
        tokenizer_name=model.default_tokenizer_name,
        max_prompt_tokens=token_budget.prompt_limit,
        enable_unanswerable_nli=source.enable_unanswerable_nli,
    )


def resolve_generation_args(args: Namespace, *, preset: str) -> Namespace:
    """Resolve generation CLI defaults from central settings."""

    generation = PROJECT_SETTINGS.generation
    eval_settings = PROJECT_SETTINGS.eval
    model = PROJECT_SETTINGS.model
    teacher = PROJECT_SETTINGS.teacher

    if preset == "structured_eval":
        defaults = {
            "split": eval_settings.structured_generation.split,
            "max_samples": eval_settings.structured_generation.max_samples,
            "seed": generation.seed,
            "base_model": model.target_training_llm,
            "include_base": eval_settings.structured_generation.include_base,
            "batch_size": eval_settings.structured_generation.batch_size,
            "max_new_tokens": eval_settings.structured_generation.max_new_tokens,
            "temperature": eval_settings.structured_generation.temperature,
            "top_p": eval_settings.structured_generation.top_p,
            "repetition_penalty": eval_settings.structured_generation.repetition_penalty,
            "prompt_mode": eval_settings.structured_generation.prompt_mode,
            "fewshot_k": eval_settings.structured_generation.fewshot_k,
            "retry_on_protocol_fail": eval_settings.structured_generation.retry_on_protocol_fail,
            "max_attempts": eval_settings.structured_generation.max_attempts,
            "use_chat_template": generation.use_chat_template,
            "strip_think_tags": generation.strip_think_tags,
            "strip_role_markers": generation.strip_role_markers,
            "record_token_usage": PROJECT_SETTINGS.token_budget.record_token_usage,
            "eval_track": eval_settings.structured_generation.eval_track,
            "eval_variant": eval_settings.structured_generation.eval_variant,
        }
        return resolve_namespace(args, **defaults)

    if preset == "flat_eval":
        defaults = {
            "split": eval_settings.flat_generation.split,
            "max_samples": eval_settings.flat_generation.max_samples,
            "seed": generation.seed,
            "base_model": model.target_training_llm,
            "include_base": eval_settings.flat_generation.include_base,
            "batch_size": eval_settings.flat_generation.batch_size,
            "max_new_tokens": eval_settings.flat_generation.max_new_tokens,
            "temperature": eval_settings.flat_generation.temperature,
            "top_p": eval_settings.flat_generation.top_p,
            "repetition_penalty": eval_settings.flat_generation.repetition_penalty,
            "use_chat_template": generation.use_chat_template,
            "strip_think_tags": generation.strip_think_tags,
            "strip_role_markers": generation.strip_role_markers,
            "encourage_refusal": eval_settings.flat_generation.encourage_refusal,
            "record_token_usage": PROJECT_SETTINGS.token_budget.record_token_usage,
            "eval_track": eval_settings.flat_generation.eval_track,
        }
        return resolve_namespace(args, **defaults)

    if preset == "teacher":
        defaults = {
            "prompt_style": teacher.prompt_style,
            "answerable_num_candidates_per_example": teacher.answerable_num_candidates_per_example,
            "unanswerable_num_candidates_per_example": teacher.unanswerable_num_candidates_per_example,
            "record_token_usage": PROJECT_SETTINGS.token_budget.record_token_usage,
        }
        return resolve_namespace(args, **defaults)

    raise ValueError(f"Unsupported generation resolver preset: {preset}")


def resolve_nli_args(args: Namespace) -> Namespace:
    """Resolve NLI and judge arguments from central settings."""

    return resolve_namespace(
        args,
        nli_model_name=PROJECT_SETTINGS.nli.model_name,
        nli_device=PROJECT_SETTINGS.nli.device,
        nli_batch_size=PROJECT_SETTINGS.nli.batch_size,
        nli_max_length=PROJECT_SETTINGS.nli.max_length,
        nli_fp16=PROJECT_SETTINGS.nli.fp16,
        temperature=PROJECT_SETTINGS.judge.temperature,
        full_margin_threshold=PROJECT_SETTINGS.judge.full_margin_threshold,
        reject_margin_threshold=PROJECT_SETTINGS.judge.reject_margin_threshold,
        reject_band_half_width=PROJECT_SETTINGS.judge.reject_band_half_width,
        qa_fail_as_negative=PROJECT_SETTINGS.judge.qa_fail_as_negative,
        qa_check_answer_type=PROJECT_SETTINGS.judge.qa_check_answer_type,
        qa_spacy_model=PROJECT_SETTINGS.judge.qa_spacy_model,
    )


def resolve_matcher_args(args: Namespace) -> Namespace:
    """Resolve matcher-related arguments from central settings."""

    return resolve_namespace(
        args,
        matcher_model_name=PROJECT_SETTINGS.matcher.model_name,
        matcher_threshold=PROJECT_SETTINGS.matcher.runtime_threshold,
    )


def resolve_training_args(args: Namespace) -> Namespace:
    """Resolve SFT dataset/training arguments from central settings."""

    training = PROJECT_SETTINGS.training
    return resolve_namespace(
        args,
        seed=training.seed,
        tokenizer_name=PROJECT_SETTINGS.model.default_tokenizer_name,
        max_train_samples=training.max_train_samples,
        max_validation_samples=training.max_validation_samples,
        max_test_samples=training.max_test_samples,
        max_train_answerable_samples=training.max_train_answerable_samples,
        max_train_unanswerable_samples=training.max_train_unanswerable_samples,
        max_validation_answerable_samples=training.max_validation_answerable_samples,
        max_validation_unanswerable_samples=training.max_validation_unanswerable_samples,
        max_test_answerable_samples=training.max_test_answerable_samples,
        max_test_unanswerable_samples=training.max_test_unanswerable_samples,
        max_prompt_tokens=training.max_prompt_tokens,
        max_completion_tokens=training.max_completion_tokens,
        model_name_or_path=PROJECT_SETTINGS.model.target_training_llm,
        train_epochs=training.train_epochs,
        learning_rate=training.learning_rate,
        train_batch_size=training.train_batch_size,
        eval_batch_size=training.eval_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        max_length=training.max_length,
        save_steps=training.save_steps,
        save_total_limit=training.save_total_limit,
        logging_steps=training.logging_steps,
        lora_r=training.lora_r,
        lora_alpha=training.lora_alpha,
        lora_dropout=training.lora_dropout,
        lora_target_modules=",".join(training.lora_target_modules),
    )


def resolve_eval_args(args: Namespace, *, preset: str) -> Namespace:
    """Resolve evaluation defaults from central settings."""

    eval_settings = PROJECT_SETTINGS.eval
    generation = PROJECT_SETTINGS.generation
    selection = eval_settings.selection

    if preset == "grounded":
        return resolve_namespace(
            args,
            split=eval_settings.grounded.split,
            max_samples=eval_settings.grounded.max_samples,
            seed=generation.seed,
            enable_semantics=eval_settings.grounded.enable_semantics,
            semantic_decision_source=eval_settings.grounded.semantic_decision_source,
            semantic_match_f1_threshold=eval_settings.grounded.semantic_match_f1_threshold,
        )
    if preset == "task_baseline":
        return resolve_namespace(
            args,
            split=eval_settings.task_baseline.split,
            max_samples=eval_settings.task_baseline.max_samples,
            seed=generation.seed,
            semantic_decision_source=eval_settings.task_baseline.semantic_decision_source,
            semantic_match_f1_threshold=eval_settings.task_baseline.semantic_match_f1_threshold,
        )
    if preset == "deepeval":
        return resolve_namespace(
            args,
            split=eval_settings.deepeval.split,
            max_samples=eval_settings.deepeval.max_samples,
            seed=generation.seed,
            input_mode=eval_settings.deepeval.input_mode,
            answer_source=eval_settings.deepeval.answer_source,
            judge_model=eval_settings.deepeval.judge_model,
            threshold=eval_settings.deepeval.threshold,
            max_concurrent=eval_settings.deepeval.max_concurrent,
            throttle_value=eval_settings.deepeval.throttle_value,
        )
    if preset == "selection":
        return resolve_namespace(
            args,
            parse_ok_threshold=selection.parse_ok_threshold,
            protocol_ok_threshold=selection.protocol_ok_threshold,
            evidence_ok_threshold=selection.evidence_ok_threshold,
        )
    if preset == "loss_curve":
        return resolve_namespace(
            args,
            split=eval_settings.loss_curve.split,
            max_samples=eval_settings.loss_curve.max_samples,
            seed=generation.seed,
            base_model=PROJECT_SETTINGS.model.target_training_llm,
            include_base=eval_settings.loss_curve.include_base,
            batch_size=eval_settings.loss_curve.batch_size,
            max_length=PROJECT_SETTINGS.training.max_length,
            eval_track=eval_settings.loss_curve.eval_track,
            eval_variant=eval_settings.loss_curve.eval_variant,
        )
    raise ValueError(f"Unsupported eval resolver preset: {preset}")


def resolve_calibration_args(args: Namespace, *, preset: str) -> Namespace:
    """Resolve calibration defaults from central settings."""

    calibration = PROJECT_SETTINGS.calibration
    generation = PROJECT_SETTINGS.generation
    nli = PROJECT_SETTINGS.nli
    if preset == "annotation_pack":
        return resolve_namespace(
            args,
            split=calibration.split,
            task_types=",".join(calibration.annotation_task_types),
            max_samples_per_task=calibration.max_samples_per_task,
            seed=generation.seed,
            pack_id="",
            metrics_out="",
        )
    if preset == "evaluate_pack":
        return resolve_namespace(
            args,
            split=calibration.split,
            max_samples=calibration.max_samples,
            seed=generation.seed,
            group_by=calibration.group_by,
            search_objective=calibration.search_objective,
            enable_nli_reject_search=calibration.enable_nli_reject_search,
            reject_alpha=calibration.reject_alpha,
            nli_model_name=nli.model_name,
            nli_device=nli.device,
            nli_batch_size=nli.batch_size,
            nli_max_length=nli.max_length,
            nli_fp16=nli.fp16,
            temperature_min=calibration.temperature_min,
            temperature_max=calibration.temperature_max,
            temperature_step=calibration.temperature_step,
            margin_threshold_values="",
            margin_threshold_min=calibration.margin_threshold_min,
            margin_threshold_max=calibration.margin_threshold_max,
            margin_threshold_step=calibration.margin_threshold_step,
            argmax_conf_threshold_values="",
            argmax_conf_threshold_min=calibration.argmax_conf_threshold_min,
            argmax_conf_threshold_max=calibration.argmax_conf_threshold_max,
            argmax_conf_threshold_step=calibration.argmax_conf_threshold_step,
            band_half_width_values="",
            band_half_width_min=calibration.band_half_width_min,
            band_half_width_max=calibration.band_half_width_max,
            band_half_width_step=calibration.band_half_width_step,
            matcher_model_name=PROJECT_SETTINGS.matcher.model_name,
            matcher_threshold_values="",
            matcher_threshold_min=calibration.matcher_threshold_min,
            matcher_threshold_max=calibration.matcher_threshold_max,
            matcher_threshold_step=calibration.matcher_threshold_step,
            summary_json="",
        )
    raise ValueError(f"Unsupported calibration resolver preset: {preset}")
