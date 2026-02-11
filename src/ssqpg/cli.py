from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import typer
from rich.console import Console

from .audit import surface_signal_audit
from .answer_filter import apply_answer_filters
from .config import (
    AnswerFilterConfig,
    AuditConfig,
    DifficultyConfig,
    GenerationConfig,
    HardAugmentConfig,
    HaluEvalSourceConfig,
    JudgeConfig,
    NLIConfig,
    PairFilterConfig,
    PairSelectConfig,
    RepairConfig,
    SourceLengthConfig,
)
from .dataset import filter_source_by_length, iter_halueval_source, iter_jsonl_source, source_to_json
from .difficulty import assign_difficulty_buckets
from .filtering import apply_pair_filters
from .generation import ApiSelfSampler, SelfSampler
from .hard_augment import repair_hard_with_api, resample_hard_answers
from .io import read_jsonl, write_json, write_jsonl
from .judge import AnswerJudge
from .nli import NLIVerifier
from .pairing import build_pairs, summarize_question_groups
from .repair import MinimalEditRepairer


app = typer.Typer(add_completion=False)
console = Console()

source_app = typer.Typer()
generate_app = typer.Typer()
judge_app = typer.Typer()
pair_app = typer.Typer()
repair_app = typer.Typer()
filter_app = typer.Typer()
audit_app = typer.Typer()
bucket_app = typer.Typer()

app.add_typer(source_app, name="source")
app.add_typer(generate_app, name="generate")
app.add_typer(judge_app, name="judge")
app.add_typer(pair_app, name="pair")
app.add_typer(repair_app, name="repair")
app.add_typer(filter_app, name="filter")
app.add_typer(audit_app, name="audit")
app.add_typer(bucket_app, name="bucket")


@app.command("version")
def version() -> None:
    console.print("ssqpg 0.1.0")


@source_app.command("sample")
def source_sample(
    out: str = typer.Option(..., help="Output source JSONL path."),
    source: str = typer.Option("halueval", help="Source type: halueval or jsonl."),
    input_jsonl: Optional[str] = typer.Option(None, help="Input JSONL path when source=jsonl."),
    max_samples: int = typer.Option(5000, help="Maximum number of sampled source records."),
    seed: int = typer.Option(42, help="Random seed."),
    include_reference_answer: bool = typer.Option(
        True,
        help="Whether to include reference_answer from source into sampled records.",
    ),
    tokenizer_name: str = typer.Option(SourceLengthConfig.tokenizer_name, help="Tokenizer for length filtering."),
    max_prompt_tokens: int = typer.Option(SourceLengthConfig.max_prompt_tokens, help="Max prompt tokens."),
    max_answer_tokens: int = typer.Option(
        SourceLengthConfig.max_answer_tokens,
        help="Max reference-answer tokens (applies only when include_reference_answer=true).",
    ),
    max_total_tokens: int = typer.Option(SourceLengthConfig.max_total_tokens, help="Max total tokens."),
    knowledge_field: str = typer.Option("knowledge", help="Knowledge field in jsonl mode."),
    question_field: str = typer.Option("question", help="Question field in jsonl mode."),
    answer_field: str = typer.Option("reference_answer", help="Reference-answer field in jsonl mode."),
    id_field: str = typer.Option("id", help="Record id field in jsonl mode."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional JSON metrics output path."),
) -> None:
    """Stage 1: sample reliable source QA records and apply length constraints."""

    source = source.strip().lower()
    if source not in {"halueval", "jsonl"}:
        raise typer.BadParameter("source must be one of: halueval, jsonl")

    if source == "halueval":
        cfg = HaluEvalSourceConfig()
        records = list(
            iter_halueval_source(
                cfg,
                max_samples=max_samples,
                seed=seed,
                include_reference_answer=include_reference_answer,
            )
        )
    else:
        if not input_jsonl:
            raise typer.BadParameter("--input-jsonl is required when source=jsonl")
        records = list(
            iter_jsonl_source(
                path=input_jsonl,
                knowledge_field=knowledge_field,
                question_field=question_field,
                answer_field=answer_field,
                id_field=id_field,
                max_samples=max_samples,
                seed=seed,
                include_reference_answer=include_reference_answer,
            )
        )

    len_cfg = SourceLengthConfig(
        tokenizer_name=tokenizer_name,
        max_prompt_tokens=max_prompt_tokens,
        max_answer_tokens=max_answer_tokens,
        max_total_tokens=max_total_tokens,
    )
    filtered, metrics = filter_source_by_length(records, len_cfg)

    rows = source_to_json(filtered, include_reference_answer=include_reference_answer)
    write_jsonl(out, rows)

    summary = {
        **metrics,
        "include_reference_answer": include_reference_answer,
    }
    if metrics_out:
        write_json(metrics_out, summary)

    console.print(f"Saved {len(rows)} source rows to {out}")
    console.print_json(json.dumps(summary))


@generate_app.command("answers")
def generate_answers(
    in_path: str = typer.Option(..., help="Input source JSONL path."),
    out: str = typer.Option(..., help="Output generated JSONL path."),
    backend: str = typer.Option(
        GenerationConfig.backend,
        help="Generation backend: local or api.",
    ),
    model_name: str = typer.Option(GenerationConfig.model_name, help="Generator model name."),
    device: str = typer.Option(GenerationConfig.device, help="Generator device."),
    batch_size: int = typer.Option(GenerationConfig.batch_size, help="Generation batch size."),
    max_new_tokens: int = typer.Option(GenerationConfig.max_new_tokens, help="Max new tokens per generation attempt."),
    max_answer_tokens: int = typer.Option(
        GenerationConfig.max_answer_tokens,
        help="Max tokens allowed for cleaned answer text.",
    ),
    max_attempts_per_sample: int = typer.Option(
        GenerationConfig.max_attempts_per_sample,
        help="Max generation attempts per sampled answer before dropping.",
    ),
    num_samples_per_question: int = typer.Option(
        GenerationConfig.num_samples_per_question,
        help="Number of sampled answers per question.",
    ),
    temperature: float = typer.Option(GenerationConfig.temperature, help="Sampling temperature."),
    top_p: float = typer.Option(GenerationConfig.top_p, help="Top-p sampling."),
    top_k: int = typer.Option(GenerationConfig.top_k, help="Top-k sampling."),
    min_p: Optional[float] = typer.Option(GenerationConfig.min_p, help="Optional min-p sampling."),
    repetition_penalty: float = typer.Option(GenerationConfig.repetition_penalty, help="Repetition penalty."),
    use_chat_template: bool = typer.Option(
        GenerationConfig.use_chat_template,
        help="Wrap prompt with tokenizer chat template before generation.",
    ),
    enable_thinking: bool = typer.Option(
        GenerationConfig.enable_thinking,
        help="Enable model thinking mode when chat template supports it.",
    ),
    strip_think_tags: bool = typer.Option(
        GenerationConfig.strip_think_tags,
        help="Strip <think>...</think> blocks from generated text.",
    ),
    strip_role_markers: bool = typer.Option(
        GenerationConfig.strip_role_markers,
        help="Strip leaked role markers (user/assistant/system).",
    ),
    strip_markdown_fences: bool = typer.Option(
        GenerationConfig.strip_markdown_fences,
        help="Strip markdown code fences from generated text.",
    ),
    extract_short_answer: bool = typer.Option(
        GenerationConfig.extract_short_answer,
        help="Extract concise answer from verbose generations.",
    ),
    include_generation_meta: bool = typer.Option(
        GenerationConfig.include_generation_meta,
        help="Include per-row generation metadata.",
    ),
    api_model_name: str = typer.Option(
        GenerationConfig.api_model_name,
        help="API model id for backend=api.",
    ),
    api_base_url: str = typer.Option(
        GenerationConfig.api_base_url,
        help="OpenAI-compatible API base URL for backend=api.",
    ),
    api_key_env: str = typer.Option(
        GenerationConfig.api_key_env,
        help="Environment variable containing API key for backend=api.",
    ),
    api_timeout_seconds: float = typer.Option(
        GenerationConfig.api_timeout_seconds,
        help="HTTP timeout (seconds) for backend=api.",
    ),
    api_max_concurrency: int = typer.Option(
        GenerationConfig.api_max_concurrency,
        help="Max concurrent API requests for backend=api.",
    ),
    api_max_retries: int = typer.Option(
        GenerationConfig.api_max_retries,
        help="Retry count for transient API failures.",
    ),
    api_retry_backoff_base: float = typer.Option(
        GenerationConfig.api_retry_backoff_base,
        help="Base exponential backoff seconds for API retries.",
    ),
    api_retry_backoff_max: float = typer.Option(
        GenerationConfig.api_retry_backoff_max,
        help="Max exponential backoff seconds for API retries.",
    ),
    seed: int = typer.Option(GenerationConfig.seed, help="Random seed."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional JSON metrics output path."),
) -> None:
    """Stage 2: self-sample multiple answers for each source QA item."""

    backend = backend.strip().lower()
    if backend not in {"local", "api"}:
        raise typer.BadParameter("backend must be one of: local, api")
    if backend == "local" and enable_thinking and not use_chat_template:
        raise typer.BadParameter("enable_thinking requires use_chat_template=true for backend=local.")

    rows = list(read_jsonl(in_path))
    cfg = GenerationConfig(
        backend=backend,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        max_answer_tokens=max_answer_tokens,
        max_attempts_per_sample=max_attempts_per_sample,
        num_samples_per_question=num_samples_per_question,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
        strip_think_tags=strip_think_tags,
        strip_role_markers=strip_role_markers,
        strip_markdown_fences=strip_markdown_fences,
        extract_short_answer=extract_short_answer,
        include_generation_meta=include_generation_meta,
        api_model_name=api_model_name,
        api_base_url=api_base_url,
        api_key_env=api_key_env,
        api_timeout_seconds=api_timeout_seconds,
        api_max_concurrency=api_max_concurrency,
        api_max_retries=api_max_retries,
        api_retry_backoff_base=api_retry_backoff_base,
        api_retry_backoff_max=api_retry_backoff_max,
        seed=seed,
    )

    if backend == "api":
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise typer.BadParameter(f"Environment variable {api_key_env} is not set.")
        sampler = ApiSelfSampler(cfg=cfg, api_key=api_key)
    else:
        sampler = SelfSampler(cfg)

    generated, gen_metrics = sampler.generate(rows)
    write_jsonl(out, generated)

    if metrics_out:
        write_json(metrics_out, gen_metrics)

    console.print(f"Saved {len(generated)} generated rows to {out}")
    console.print_json(json.dumps(gen_metrics))


@judge_app.command("answers")
def judge_answers(
    in_path: str = typer.Option(..., help="Input generated JSONL path."),
    out: str = typer.Option(..., help="Output judged JSONL path."),
    primary_nli_model: str = typer.Option(NLIConfig.model_name, help="Primary NLI model name."),
    secondary_nli_model: Optional[str] = typer.Option(None, help="Optional secondary NLI model name."),
    device: str = typer.Option(NLIConfig.device, help="NLI device."),
    batch_size: int = typer.Option(NLIConfig.batch_size, help="NLI batch size."),
    max_length: int = typer.Option(NLIConfig.max_length, help="NLI max length."),
    fp16: bool = typer.Option(NLIConfig.fp16, help="Whether to enable fp16 for NLI on CUDA."),
    reference_entail_threshold: float = typer.Option(JudgeConfig.reference_entail_threshold, help="Reference entail threshold."),
    candidate_entail_threshold: float = typer.Option(JudgeConfig.candidate_entail_threshold, help="Candidate entail threshold."),
    candidate_contradict_threshold: float = typer.Option(
        JudgeConfig.candidate_contradict_threshold,
        help="Candidate contradiction upper threshold.",
    ),
    vote_mode: str = typer.Option(JudgeConfig.vote_mode, help="primary | and | or"),
    qa_similarity_model_name: Optional[str] = typer.Option(
        JudgeConfig.qa_similarity_model_name,
        help="Sentence-Transformer model for QA consistency. Use empty string to disable.",
    ),
    qa_similarity_min: float = typer.Option(JudgeConfig.qa_similarity_min, help="QA similarity threshold."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional JSON metrics output path."),
) -> None:
    """Stage 3: judge sampled answers with NLI support and QA consistency."""

    rows = list(read_jsonl(in_path))
    primary = NLIVerifier(
        model_name=primary_nli_model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        fp16=fp16,
    )

    secondary = None
    if secondary_nli_model:
        secondary = NLIVerifier(
            model_name=secondary_nli_model,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            fp16=fp16,
        )

    qa_model_name = qa_similarity_model_name or None
    cfg = JudgeConfig(
        reference_entail_threshold=reference_entail_threshold,
        candidate_entail_threshold=candidate_entail_threshold,
        candidate_contradict_threshold=candidate_contradict_threshold,
        vote_mode=vote_mode,
        qa_similarity_model_name=qa_model_name,
        qa_similarity_min=qa_similarity_min,
    )
    judge = AnswerJudge(cfg=cfg, primary_verifier=primary, secondary_verifier=secondary)
    judged, metrics = judge.judge(rows)
    write_jsonl(out, judged)

    if metrics_out:
        write_json(metrics_out, metrics)

    console.print(f"Saved {len(judged)} judged rows to {out}")
    console.print_json(json.dumps(metrics))


@pair_app.command("build")
def pair_build(
    in_path: str = typer.Option(..., help="Input judged JSONL path."),
    out: str = typer.Option(..., help="Output pair-level JSONL path."),
    out_filtered_judged: Optional[str] = typer.Option(None, help="Optional output path for answer-filtered judged rows."),
    out_augmented_judged: Optional[str] = typer.Option(None, help="Optional output path for augmented judged rows."),
    tokenizer_name: str = typer.Option(AnswerFilterConfig.tokenizer_name, help="Tokenizer name for answer-token filtering."),
    min_answer_tokens: int = typer.Option(AnswerFilterConfig.min_answer_tokens, help="Minimum answer tokens."),
    max_answer_tokens: int = typer.Option(AnswerFilterConfig.max_answer_tokens, help="Maximum answer tokens."),
    max_answer_chars: int = typer.Option(AnswerFilterConfig.max_answer_chars, help="Maximum answer characters."),
    require_qa_consistent: bool = typer.Option(AnswerFilterConfig.require_qa_consistent, help="Require qa_consistent=true from judged rows."),
    require_reference_supported: bool = typer.Option(
        AnswerFilterConfig.require_reference_supported,
        help="Require reference_supported_primary=true when reference exists.",
    ),
    drop_prompt_leak: bool = typer.Option(AnswerFilterConfig.drop_prompt_leak, help="Drop answers with prompt-leak markers."),
    drop_option_style: bool = typer.Option(AnswerFilterConfig.drop_option_style, help="Drop option-style answers."),
    drop_easy: bool = typer.Option(PairSelectConfig.drop_easy, help="Drop easy questions after grouping."),
    keep_unresolved_hard: bool = typer.Option(PairSelectConfig.keep_unresolved_hard, help="Keep unresolved hard questions as needs_positive."),
    hard_negative_entail_weight: float = typer.Option(
        PairSelectConfig.hard_negative_entail_weight,
        help="Weight of rejected entailment when selecting hard negatives.",
    ),
    hard_negative_edit_proximity_weight: float = typer.Option(
        PairSelectConfig.hard_negative_edit_proximity_weight,
        help="Weight of edit proximity when selecting hard negatives.",
    ),
    hard_negative_length_proximity_weight: float = typer.Option(
        PairSelectConfig.hard_negative_length_proximity_weight,
        help="Weight of length proximity when selecting hard negatives.",
    ),
    extra_samples_per_hard: int = typer.Option(HardAugmentConfig.extra_samples_per_hard, help="Extra same-model samples per hard question."),
    resample_backend: str = typer.Option(HardAugmentConfig.resample_backend, help="Resampling backend: local or api."),
    resample_model_name: str = typer.Option(HardAugmentConfig.resample_model_name, help="Resampling model name for backend=local."),
    resample_device: str = typer.Option(HardAugmentConfig.resample_device, help="Resampling device for backend=local."),
    resample_batch_size: int = typer.Option(HardAugmentConfig.resample_batch_size, help="Resampling batch size."),
    resample_max_new_tokens: int = typer.Option(HardAugmentConfig.resample_max_new_tokens, help="Resampling max new tokens."),
    resample_temperature: float = typer.Option(HardAugmentConfig.resample_temperature, help="Resampling temperature."),
    resample_top_p: float = typer.Option(HardAugmentConfig.resample_top_p, help="Resampling top-p."),
    resample_top_k: int = typer.Option(HardAugmentConfig.resample_top_k, help="Resampling top-k."),
    resample_min_p: Optional[float] = typer.Option(HardAugmentConfig.resample_min_p, help="Resampling min-p."),
    resample_repetition_penalty: float = typer.Option(
        HardAugmentConfig.resample_repetition_penalty,
        help="Resampling repetition penalty.",
    ),
    resample_api_model_name: str = typer.Option(HardAugmentConfig.resample_api_model_name, help="Resampling API model name."),
    resample_api_base_url: str = typer.Option(HardAugmentConfig.resample_api_base_url, help="Resampling API base URL."),
    resample_api_key_env: str = typer.Option(HardAugmentConfig.resample_api_key_env, help="Resampling API key env name."),
    resample_api_timeout_seconds: float = typer.Option(HardAugmentConfig.resample_api_timeout_seconds, help="Resampling API timeout seconds."),
    resample_api_max_concurrency: int = typer.Option(HardAugmentConfig.resample_api_max_concurrency, help="Resampling API max concurrency."),
    resample_api_max_retries: int = typer.Option(HardAugmentConfig.resample_api_max_retries, help="Resampling API max retries."),
    enable_api_repair: bool = typer.Option(HardAugmentConfig.enable_api_repair, help="Enable strong-model API repair on unresolved hard questions."),
    repair_api_model_name: str = typer.Option(HardAugmentConfig.repair_api_model_name, help="Strong-model API name for hard repair."),
    repair_api_base_url: str = typer.Option(HardAugmentConfig.repair_api_base_url, help="Strong-model API base URL."),
    repair_api_key_env: str = typer.Option(HardAugmentConfig.repair_api_key_env, help="Strong-model API key env name."),
    repair_api_timeout_seconds: float = typer.Option(HardAugmentConfig.repair_api_timeout_seconds, help="Strong-model API timeout seconds."),
    repair_api_max_concurrency: int = typer.Option(HardAugmentConfig.repair_api_max_concurrency, help="Strong-model API max concurrency."),
    repair_api_max_retries: int = typer.Option(HardAugmentConfig.repair_api_max_retries, help="Strong-model API max retries."),
    repair_attempts_per_question: int = typer.Option(HardAugmentConfig.repair_attempts_per_question, help="Repair attempts per unresolved hard question."),
    repair_max_new_tokens: int = typer.Option(HardAugmentConfig.repair_max_new_tokens, help="Repair max new tokens."),
    repair_temperature: float = typer.Option(HardAugmentConfig.repair_temperature, help="Repair temperature."),
    repair_top_p: float = typer.Option(HardAugmentConfig.repair_top_p, help="Repair top-p."),
    primary_nli_model: str = typer.Option(NLIConfig.model_name, help="Primary NLI model for judging augmented answers."),
    secondary_nli_model: Optional[str] = typer.Option(None, help="Optional secondary NLI model for judging augmented answers."),
    nli_device: str = typer.Option(NLIConfig.device, help="NLI device for augmentation judgment."),
    nli_batch_size: int = typer.Option(NLIConfig.batch_size, help="NLI batch size for augmentation judgment."),
    nli_max_length: int = typer.Option(NLIConfig.max_length, help="NLI max length for augmentation judgment."),
    nli_fp16: bool = typer.Option(NLIConfig.fp16, help="Whether to enable fp16 for augmentation NLI."),
    reference_entail_threshold: float = typer.Option(JudgeConfig.reference_entail_threshold, help="Reference entail threshold."),
    candidate_entail_threshold: float = typer.Option(JudgeConfig.candidate_entail_threshold, help="Candidate entail threshold."),
    candidate_contradict_threshold: float = typer.Option(JudgeConfig.candidate_contradict_threshold, help="Candidate contradiction threshold."),
    vote_mode: str = typer.Option(JudgeConfig.vote_mode, help="Vote mode: primary | and | or."),
    qa_similarity_model_name: Optional[str] = typer.Option(
        JudgeConfig.qa_similarity_model_name,
        help="Sentence-Transformer model for QA consistency in augmentation judge. Use empty string to disable.",
    ),
    qa_similarity_min: float = typer.Option(JudgeConfig.qa_similarity_min, help="QA similarity threshold in augmentation judge."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional JSON metrics output path."),
) -> None:
    """Stage 4: answer-filter -> regroup -> hard augmentation -> pair selection."""

    rows = list(read_jsonl(in_path))

    answer_cfg = AnswerFilterConfig(
        tokenizer_name=tokenizer_name,
        min_answer_tokens=min_answer_tokens,
        max_answer_tokens=max_answer_tokens,
        max_answer_chars=max_answer_chars,
        require_qa_consistent=require_qa_consistent,
        require_reference_supported=require_reference_supported,
        drop_prompt_leak=drop_prompt_leak,
        drop_option_style=drop_option_style,
    )
    filtered_rows, answer_filter_metrics = apply_answer_filters(rows, answer_cfg)
    if out_filtered_judged:
        write_jsonl(out_filtered_judged, filtered_rows)

    _, group_counts_initial = summarize_question_groups(filtered_rows)

    need_augment = bool(extra_samples_per_hard > 0 or enable_api_repair)
    judge: Optional[AnswerJudge] = None
    if need_augment:
        primary = NLIVerifier(
            model_name=primary_nli_model,
            device=nli_device,
            batch_size=nli_batch_size,
            max_length=nli_max_length,
            fp16=nli_fp16,
        )
        secondary = None
        if secondary_nli_model:
            secondary = NLIVerifier(
                model_name=secondary_nli_model,
                device=nli_device,
                batch_size=nli_batch_size,
                max_length=nli_max_length,
                fp16=nli_fp16,
            )
        qa_model_name = qa_similarity_model_name or None
        judge_cfg = JudgeConfig(
            reference_entail_threshold=reference_entail_threshold,
            candidate_entail_threshold=candidate_entail_threshold,
            candidate_contradict_threshold=candidate_contradict_threshold,
            vote_mode=vote_mode,
            qa_similarity_model_name=qa_model_name,
            qa_similarity_min=qa_similarity_min,
        )
        judge = AnswerJudge(cfg=judge_cfg, primary_verifier=primary, secondary_verifier=secondary)

    hard_cfg = HardAugmentConfig(
        extra_samples_per_hard=extra_samples_per_hard,
        resample_backend=resample_backend,
        resample_model_name=resample_model_name,
        resample_device=resample_device,
        resample_batch_size=resample_batch_size,
        resample_max_new_tokens=resample_max_new_tokens,
        resample_temperature=resample_temperature,
        resample_top_p=resample_top_p,
        resample_top_k=resample_top_k,
        resample_min_p=resample_min_p,
        resample_repetition_penalty=resample_repetition_penalty,
        resample_api_model_name=resample_api_model_name,
        resample_api_base_url=resample_api_base_url,
        resample_api_key_env=resample_api_key_env,
        resample_api_timeout_seconds=resample_api_timeout_seconds,
        resample_api_max_concurrency=resample_api_max_concurrency,
        resample_api_max_retries=resample_api_max_retries,
        enable_api_repair=enable_api_repair,
        repair_api_model_name=repair_api_model_name,
        repair_api_base_url=repair_api_base_url,
        repair_api_key_env=repair_api_key_env,
        repair_api_timeout_seconds=repair_api_timeout_seconds,
        repair_api_max_concurrency=repair_api_max_concurrency,
        repair_api_max_retries=repair_api_max_retries,
        repair_attempts_per_question=repair_attempts_per_question,
        repair_max_new_tokens=repair_max_new_tokens,
        repair_temperature=repair_temperature,
        repair_top_p=repair_top_p,
    )

    working_rows = list(filtered_rows)
    resampled_rows: List[Dict[str, Any]] = []
    resample_metrics: Dict[str, Any] = {"skipped": "augmentation_disabled"}
    repaired_rows: List[Dict[str, Any]] = []
    repair_metrics: Dict[str, Any] = {"skipped": "augmentation_disabled"}
    group_counts_after_resample = dict(group_counts_initial)

    if judge is not None:
        resampled_rows, resample_metrics = resample_hard_answers(
            rows=working_rows,
            hard_cfg=hard_cfg,
            answer_filter_cfg=answer_cfg,
            judge=judge,
        )
        working_rows.extend(resampled_rows)
        _, group_counts_after_resample = summarize_question_groups(working_rows)

        repaired_rows, repair_metrics = repair_hard_with_api(
            rows=working_rows,
            hard_cfg=hard_cfg,
            answer_filter_cfg=answer_cfg,
            judge=judge,
        )
        working_rows.extend(repaired_rows)

    dedup: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for row in working_rows:
        key = (str(row["id"]), int(row.get("sample_id", -1)), str(row.get("answer") or ""))
        dedup[key] = row
    working_rows = list(dedup.values())

    _, group_counts_final = summarize_question_groups(working_rows)
    if out_augmented_judged:
        write_jsonl(out_augmented_judged, working_rows)

    pair_cfg = PairSelectConfig(
        drop_easy=drop_easy,
        keep_unresolved_hard=keep_unresolved_hard,
        hard_negative_entail_weight=hard_negative_entail_weight,
        hard_negative_edit_proximity_weight=hard_negative_edit_proximity_weight,
        hard_negative_length_proximity_weight=hard_negative_length_proximity_weight,
    )
    pairs, status_counts, pair_group_counts = build_pairs(working_rows, pair_cfg)
    write_jsonl(out, pairs)

    metrics = {
        "num_input_rows": len(rows),
        "answer_filter_metrics": answer_filter_metrics,
        "group_counts_initial": group_counts_initial,
        "resample_metrics": resample_metrics,
        "num_resampled_rows_kept": len(resampled_rows),
        "group_counts_after_resample": group_counts_after_resample,
        "repair_metrics": repair_metrics,
        "num_repaired_rows_kept": len(repaired_rows),
        "group_counts_final": group_counts_final,
        "num_augmented_judged_rows": len(working_rows),
        "num_pair_rows": len(pairs),
        "pair_group_counts": pair_group_counts,
        "pair_status_counts": status_counts,
    }
    if metrics_out:
        write_json(metrics_out, metrics)

    console.print(f"Saved {len(pairs)} pair rows to {out}")
    console.print_json(json.dumps(metrics))


@repair_app.command("hard")
def repair_hard(
    in_path: str = typer.Option(..., help="Input pair-level JSONL path."),
    out: str = typer.Option(..., help="Output repaired pair-level JSONL path."),
    repair_model_name: str = typer.Option(RepairConfig.model_name, help="Repair model name."),
    repair_device: str = typer.Option(RepairConfig.device, help="Repair model device."),
    repair_batch_size: int = typer.Option(RepairConfig.batch_size, help="Repair generation batch size."),
    max_new_tokens: int = typer.Option(RepairConfig.max_new_tokens, help="Max new tokens for repair generation."),
    attempts_per_record: int = typer.Option(RepairConfig.attempts_per_record, help="Repair attempts per hard record."),
    temperature: float = typer.Option(RepairConfig.temperature, help="Repair sampling temperature."),
    top_p: float = typer.Option(RepairConfig.top_p, help="Repair top-p."),
    top_k: int = typer.Option(RepairConfig.top_k, help="Repair top-k."),
    repetition_penalty: float = typer.Option(RepairConfig.repetition_penalty, help="Repair repetition penalty."),
    entail_threshold: float = typer.Option(RepairConfig.entail_threshold, help="Repair NLI entail threshold."),
    min_norm_edit: float = typer.Option(RepairConfig.min_norm_edit, help="Min normalized edit vs rejected."),
    max_norm_edit: float = typer.Option(RepairConfig.max_norm_edit, help="Max normalized edit vs rejected."),
    min_length_ratio: float = typer.Option(RepairConfig.min_length_ratio, help="Min repaired/rejected length ratio."),
    max_length_ratio: float = typer.Option(RepairConfig.max_length_ratio, help="Max repaired/rejected length ratio."),
    fallback_to_reference: bool = typer.Option(RepairConfig.fallback_to_reference, help="Fallback to reference if repair fails."),
    seed: int = typer.Option(RepairConfig.seed, help="Repair random seed."),
    nli_model: str = typer.Option(NLIConfig.model_name, help="NLI model for repair validation."),
    nli_device: str = typer.Option(NLIConfig.device, help="NLI device for repair validation."),
    nli_batch_size: int = typer.Option(NLIConfig.batch_size, help="NLI batch size for repair validation."),
    nli_max_length: int = typer.Option(NLIConfig.max_length, help="NLI max length for repair validation."),
    nli_fp16: bool = typer.Option(NLIConfig.fp16, help="Whether to enable fp16 for repair NLI on CUDA."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional JSON metrics output path."),
) -> None:
    """Stage 5: repair hard records with minimal edits validated by NLI."""

    rows = list(read_jsonl(in_path))
    verifier = NLIVerifier(
        model_name=nli_model,
        device=nli_device,
        batch_size=nli_batch_size,
        max_length=nli_max_length,
        fp16=nli_fp16,
    )
    cfg = RepairConfig(
        model_name=repair_model_name,
        device=repair_device,
        batch_size=repair_batch_size,
        max_new_tokens=max_new_tokens,
        attempts_per_record=attempts_per_record,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        entail_threshold=entail_threshold,
        min_norm_edit=min_norm_edit,
        max_norm_edit=max_norm_edit,
        min_length_ratio=min_length_ratio,
        max_length_ratio=max_length_ratio,
        fallback_to_reference=fallback_to_reference,
        seed=seed,
    )
    repairer = MinimalEditRepairer(cfg=cfg, verifier=verifier)
    repaired_rows, metrics = repairer.repair(rows)
    write_jsonl(out, repaired_rows)

    if metrics_out:
        write_json(metrics_out, metrics)

    console.print(f"Saved {len(repaired_rows)} repaired rows to {out}")
    console.print_json(json.dumps(metrics))


@filter_app.command("apply")
def filter_apply(
    in_path: str = typer.Option(..., help="Input pair-level JSONL path."),
    out: str = typer.Option(..., help="Output final DPO pair JSONL path."),
    nli_model: str = typer.Option(NLIConfig.model_name, help="NLI model for final filtering."),
    nli_device: str = typer.Option(NLIConfig.device, help="NLI device for final filtering."),
    nli_batch_size: int = typer.Option(NLIConfig.batch_size, help="NLI batch size for final filtering."),
    nli_max_length: int = typer.Option(NLIConfig.max_length, help="NLI max length for final filtering."),
    nli_fp16: bool = typer.Option(NLIConfig.fp16, help="Whether to enable fp16 for final NLI on CUDA."),
    entail_threshold_pos: float = typer.Option(0.60, help="Chosen entailment threshold."),
    entail_threshold_neg: float = typer.Option(0.35, help="Rejected entailment threshold."),
    min_norm_edit: float = typer.Option(PairFilterConfig.min_norm_edit, help="Min normalized edit distance."),
    max_norm_edit: float = typer.Option(PairFilterConfig.max_norm_edit, help="Max normalized edit distance."),
    min_length_ratio: float = typer.Option(PairFilterConfig.min_length_ratio, help="Min rejected/chosen length ratio."),
    max_length_ratio: float = typer.Option(PairFilterConfig.max_length_ratio, help="Max rejected/chosen length ratio."),
    enforce_answer_type: bool = typer.Option(PairFilterConfig.enforce_answer_type, help="Enable answer-type filter."),
    spacy_model: str = typer.Option(PairFilterConfig.spacy_model, help="spaCy model for answer-type filter."),
    qa_similarity_model_name: Optional[str] = typer.Option(
        PairFilterConfig.qa_similarity_model_name,
        help="Sentence-Transformer model for QA consistency. Use empty string to disable.",
    ),
    qa_similarity_min: float = typer.Option(PairFilterConfig.qa_similarity_min, help="QA consistency similarity threshold."),
    include_prompt: bool = typer.Option(
        False,
        help="Whether to include prompt field in final exported pairs.",
    ),
    metrics_out: Optional[str] = typer.Option(None, help="Optional JSON metrics output path."),
) -> None:
    """Stage 6: final pair filtering and export for downstream DPO preprocessing."""

    rows = list(read_jsonl(in_path))
    verifier = NLIVerifier(
        model_name=nli_model,
        device=nli_device,
        batch_size=nli_batch_size,
        max_length=nli_max_length,
        fp16=nli_fp16,
    )
    qa_model_name = qa_similarity_model_name or None
    cfg = PairFilterConfig(
        min_norm_edit=min_norm_edit,
        max_norm_edit=max_norm_edit,
        min_length_ratio=min_length_ratio,
        max_length_ratio=max_length_ratio,
        enforce_answer_type=enforce_answer_type,
        spacy_model=spacy_model,
        qa_similarity_model_name=qa_model_name,
        qa_similarity_min=qa_similarity_min,
    )
    kept, metrics = apply_pair_filters(
        rows=rows,
        cfg=cfg,
        verifier=verifier,
        entail_threshold_pos=entail_threshold_pos,
        entail_threshold_neg=entail_threshold_neg,
        include_prompt=include_prompt,
    )
    write_jsonl(out, kept)

    if metrics_out:
        write_json(metrics_out, metrics)

    console.print(f"Saved {len(kept)} final pairs to {out}")
    console.print_json(json.dumps(metrics))


@audit_app.command("surface")
def audit_surface(
    in_path: str = typer.Option(..., help="Input final pair JSONL path."),
    out: str = typer.Option(..., help="Output audit JSON path."),
    separability_threshold: float = typer.Option(AuditConfig.separability_threshold, help="Flag threshold for feature separability."),
) -> None:
    """Stage 7: run a lightweight surface-signal separability audit."""

    rows = list(read_jsonl(in_path))
    cfg = AuditConfig(separability_threshold=separability_threshold)
    report = surface_signal_audit(rows, cfg)
    write_json(out, report)

    console.print(f"Saved audit report to {out}")
    console.print_json(json.dumps(report))


@bucket_app.command("difficulty")
def bucket_difficulty(
    in_path: str = typer.Option(..., help="Input final pair JSONL path."),
    out: str = typer.Option(..., help="Output pair JSONL path with difficulty buckets."),
    model_name: str = typer.Option(DifficultyConfig.model_name, help="Model used for log-prob scoring."),
    device: str = typer.Option(DifficultyConfig.device, help="Device for difficulty scoring."),
    batch_size: int = typer.Option(DifficultyConfig.batch_size, help="Batch size for difficulty scoring."),
    max_length: int = typer.Option(DifficultyConfig.max_length, help="Max sequence length for difficulty scoring."),
    fp16: bool = typer.Option(DifficultyConfig.fp16, help="Whether to enable fp16 for difficulty scoring on CUDA."),
    hard_max_delta: float = typer.Option(DifficultyConfig.hard_max_delta, help="Hard bucket upper bound for delta."),
    medium_max_delta: float = typer.Option(DifficultyConfig.medium_max_delta, help="Medium bucket upper bound for delta."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional JSON metrics output path."),
) -> None:
    """Stage 8: assign easy/medium/hard buckets using policy log-prob gaps."""

    rows = list(read_jsonl(in_path))
    cfg = DifficultyConfig(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        fp16=fp16,
        hard_max_delta=hard_max_delta,
        medium_max_delta=medium_max_delta,
    )
    out_rows, counts = assign_difficulty_buckets(rows=rows, cfg=cfg)
    write_jsonl(out, out_rows)

    summary = {"difficulty_bucket_counts": counts, "num_rows": len(out_rows)}
    if metrics_out:
        write_json(metrics_out, summary)

    console.print(f"Saved {len(out_rows)} rows with difficulty buckets to {out}")
    console.print_json(json.dumps(summary))


if __name__ == "__main__":
    app()
