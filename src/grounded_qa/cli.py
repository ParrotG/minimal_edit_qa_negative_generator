from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console

from qa_checks.correctness import CorrectnessConfig
from qa_data import ConstructionConfig, HotpotSourceConfig, NegativeSamplingConfig, SplitConfig

from .config import (
    SftRecordConfig,
    TeacherGenerationConfig,
    UnanswerablePipelineConfig,
    UnanswerablePrefilterConfig,
    ValidationConfig,
)
from .workflow import (
    build_hotpot_source,
    build_sft_records,
    build_simple_negatives,
    build_unanswerable_source,
    export_split_hotpot_rows,
    generate_teacher_candidates,
    prefilter_unanswerable_source,
    split_examples,
    validate_teacher_candidates,
    validate_unanswerable_teacher_candidates,
)


app = typer.Typer(add_completion=False)
console = Console()

source_app = typer.Typer()
teacher_app = typer.Typer()
build_app = typer.Typer()

app.add_typer(source_app, name="source")
app.add_typer(teacher_app, name="teacher")
app.add_typer(build_app, name="build")


@app.command("version")
def version() -> None:
    """Print the grounded QA pipeline version."""

    console.print("grounded_qa 0.1.0")


@source_app.command("hotpot")
def source_hotpot(
    out: str = typer.Option(..., help="Output raw grounded-QA JSONL path."),
    split: str = typer.Option(HotpotSourceConfig.split, help="HotpotQA split."),
    dataset_name: str = typer.Option(HotpotSourceConfig.dataset_name, help="HotpotQA dataset name."),
    max_samples: int = typer.Option(HotpotSourceConfig.max_samples, help="Maximum source rows to load. -1 keeps all."),
    seed: int = typer.Option(HotpotSourceConfig.seed, help="Random seed."),
    window_size: int = typer.Option(ConstructionConfig.window_size, help="Support window size."),
    max_supporting_facts: int = typer.Option(ConstructionConfig.max_supporting_facts, help="Maximum supporting facts."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional metrics JSON path."),
) -> None:
    """Build answerable grounded-QA raw examples from HotpotQA."""

    metrics = build_hotpot_source(
        out_path=out,
        metrics_out=metrics_out,
        source_cfg=HotpotSourceConfig(
            dataset_name=dataset_name,
            split=split,
            max_samples=max_samples,
            seed=seed,
        ),
        construct_cfg=ConstructionConfig(
            window_size=window_size,
            max_supporting_facts=max_supporting_facts,
        ),
    )
    console.print(f"Saved grounded-QA raw examples to {out}")
    console.print_json(json.dumps(metrics))


@source_app.command("negatives")
def source_negatives(
    in_path: str = typer.Option(..., help="Input answerable raw example JSONL path."),
    out: str = typer.Option(..., help="Output unanswerable raw example JSONL path."),
    drop_supporting_facts: int = typer.Option(NegativeSamplingConfig.drop_supporting_facts, help="Number of support facts to drop."),
    max_same_doc_sentences: int = typer.Option(NegativeSamplingConfig.max_same_doc_sentences, help="Number of same-document distractor sentences."),
    max_adjacent_doc_sentences: int = typer.Option(NegativeSamplingConfig.max_adjacent_doc_sentences, help="Number of adjacent-document distractor sentences."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional metrics JSON path."),
) -> None:
    """Derive legacy simple unanswerable examples."""

    metrics = build_simple_negatives(
        in_path=in_path,
        out_path=out,
        metrics_out=metrics_out,
        negative_cfg=NegativeSamplingConfig(
            drop_supporting_facts=drop_supporting_facts,
            max_same_doc_sentences=max_same_doc_sentences,
            max_adjacent_doc_sentences=max_adjacent_doc_sentences,
        ),
    )
    console.print(f"Saved simple negatives to {out}")
    console.print_json(json.dumps(metrics))


@source_app.command("hotpot-raw")
def source_hotpot_raw(
    out: str = typer.Option(..., help="Output split-assigned raw Hotpot JSONL path."),
    split: str = typer.Option(HotpotSourceConfig.split, help="HotpotQA split."),
    dataset_name: str = typer.Option(HotpotSourceConfig.dataset_name, help="HotpotQA dataset name."),
    max_samples: int = typer.Option(HotpotSourceConfig.max_samples, help="Maximum source rows to load. -1 keeps all."),
    seed: int = typer.Option(HotpotSourceConfig.seed, help="Random seed."),
    validation_ratio: float = typer.Option(SplitConfig.validation_ratio, help="Validation split ratio."),
    test_ratio: float = typer.Option(SplitConfig.test_ratio, help="Test split ratio."),
    train_sft_ratio: float = typer.Option(SplitConfig.train_sft_ratio, help="Train-SFT split ratio."),
    train_dpo_ratio: float = typer.Option(SplitConfig.train_dpo_ratio, help="Train-DPO split ratio."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional metrics JSON path."),
) -> None:
    """Export raw Hotpot rows with stable split assignments."""

    metrics = export_split_hotpot_rows(
        out_path=out,
        metrics_out=metrics_out,
        source_cfg=HotpotSourceConfig(
            dataset_name=dataset_name,
            split=split,
            max_samples=max_samples,
            seed=seed,
        ),
        split_cfg=SplitConfig(
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            train_sft_ratio=train_sft_ratio,
            train_dpo_ratio=train_dpo_ratio,
        ),
    )
    console.print(f"Saved split-assigned raw Hotpot rows to {out}")
    console.print_json(json.dumps(metrics))


@source_app.command("unanswerable")
def source_unanswerable(
    paired_answerable_path: str = typer.Option(..., help="Input answerable raw JSONL path."),
    raw_hotpot_pool_path: str = typer.Option(..., help="Input split-assigned raw Hotpot JSONL path."),
    out: str = typer.Option(..., help="Output unanswerable raw JSONL path."),
    target_split: str = typer.Option(UnanswerablePipelineConfig.target_split, help="Target split to sample from."),
    paired_fraction: float = typer.Option(UnanswerablePipelineConfig.paired_fraction, help="Approximate paired-source fraction."),
    max_total_examples: int = typer.Option(UnanswerablePipelineConfig.max_total_examples, help="Approximate total output count. -1 keeps all paired and matches external count."),
    replace_supporting_facts_min: int = typer.Option(UnanswerablePipelineConfig.replace_supporting_facts_min, help="Minimum number of supporting facts to replace."),
    replace_supporting_facts_max: int = typer.Option(UnanswerablePipelineConfig.replace_supporting_facts_max, help="Maximum number of supporting facts to replace."),
    same_doc_candidate_radius: int = typer.Option(UnanswerablePipelineConfig.same_doc_candidate_radius, help="Same-document neighbor radius for replacement sentences."),
    allow_same_doc_non_adjacent: bool = typer.Option(UnanswerablePipelineConfig.allow_same_doc_non_adjacent, help="Allow fallback to same-document non-adjacent sentences."),
    adjacent_doc_sentence_limit: int = typer.Option(UnanswerablePipelineConfig.adjacent_doc_sentence_limit, help="Maximum number of sentences to consider from adjacent documents."),
    window_size: int = typer.Option(ConstructionConfig.window_size, help="Support window size used when external raw rows are scaffolded into answerable examples."),
    max_supporting_facts: int = typer.Option(ConstructionConfig.max_supporting_facts, help="Maximum supporting facts when scaffold-building external raw rows."),
    include_title_prefix: bool = typer.Option(UnanswerablePipelineConfig.include_title_prefix, help="Whether to prefix replacement blocks with document titles."),
    seed: int = typer.Option(UnanswerablePipelineConfig.seed, help="Random seed."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional metrics JSON path."),
) -> None:
    """Build v1 unanswerable raw examples from paired and external pools."""

    metrics = build_unanswerable_source(
        paired_answerable_path=paired_answerable_path,
        raw_hotpot_pool_path=raw_hotpot_pool_path,
        out_path=out,
        metrics_out=metrics_out,
        pipeline_cfg=UnanswerablePipelineConfig(
            target_split=target_split,
            paired_fraction=paired_fraction,
            max_total_examples=max_total_examples,
            replace_supporting_facts_min=replace_supporting_facts_min,
            replace_supporting_facts_max=replace_supporting_facts_max,
            same_doc_candidate_radius=same_doc_candidate_radius,
            allow_same_doc_non_adjacent=allow_same_doc_non_adjacent,
            adjacent_doc_sentence_limit=adjacent_doc_sentence_limit,
            include_title_prefix=include_title_prefix,
            seed=seed,
        ),
        construct_cfg=ConstructionConfig(
            window_size=window_size,
            max_supporting_facts=max_supporting_facts,
            include_title_prefix=include_title_prefix,
        ),
    )
    console.print(f"Saved v1 unanswerable raw rows to {out}")
    console.print_json(json.dumps(metrics))


@source_app.command("unanswerable-prefilter")
def source_unanswerable_prefilter(
    in_path: str = typer.Option(..., help="Input unanswerable raw JSONL path."),
    out: str = typer.Option(..., help="Output NLI-prefiltered unanswerable raw JSONL path."),
    enable_nli_prefilter: bool = typer.Option(UnanswerablePrefilterConfig.enable_nli_prefilter, help="Enable the unanswerable NLI prefilter."),
    judge_decision_source: str = typer.Option(UnanswerablePrefilterConfig.judge_decision_source, help="Decision source used for prefiltering. Only full_binary is supported."),
    nli_model_name: str = typer.Option(UnanswerablePrefilterConfig.nli_model_name, help="NLI model name."),
    nli_device: str = typer.Option(UnanswerablePrefilterConfig.nli_device, help="NLI device."),
    nli_batch_size: int = typer.Option(UnanswerablePrefilterConfig.nli_batch_size, help="NLI batch size."),
    nli_max_length: int = typer.Option(UnanswerablePrefilterConfig.nli_max_length, help="NLI max length."),
    nli_fp16: bool = typer.Option(UnanswerablePrefilterConfig.nli_fp16, help="Whether to use fp16 for the NLI verifier."),
    temperature: float = typer.Option(UnanswerablePrefilterConfig.temperature, help="Temperature used by the calibrated NLI judge."),
    full_margin_threshold: float = typer.Option(UnanswerablePrefilterConfig.full_margin_threshold, help="Full-binary margin threshold."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional metrics JSON path."),
) -> None:
    """Prefilter unanswerable raw rows using NLI against the original reference answer."""

    metrics = prefilter_unanswerable_source(
        in_path=in_path,
        out_path=out,
        metrics_out=metrics_out,
        cfg=UnanswerablePrefilterConfig(
            enable_nli_prefilter=enable_nli_prefilter,
            judge_decision_source=judge_decision_source,
            nli_model_name=nli_model_name,
            nli_device=nli_device,
            nli_batch_size=nli_batch_size,
            nli_max_length=nli_max_length,
            nli_fp16=nli_fp16,
            temperature=temperature,
            full_margin_threshold=full_margin_threshold,
        ),
    )
    console.print(f"Saved NLI-prefiltered unanswerable rows to {out}")
    console.print_json(json.dumps(metrics))


@source_app.command("split")
def source_split(
    in_path: str = typer.Option(..., help="Input raw example JSONL path."),
    out: str = typer.Option(..., help="Output split-assigned JSONL path."),
    validation_ratio: float = typer.Option(SplitConfig.validation_ratio, help="Validation split ratio."),
    test_ratio: float = typer.Option(SplitConfig.test_ratio, help="Test split ratio."),
    train_sft_ratio: float = typer.Option(SplitConfig.train_sft_ratio, help="Train-SFT split ratio."),
    train_dpo_ratio: float = typer.Option(SplitConfig.train_dpo_ratio, help="Train-DPO split ratio."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional metrics JSON path."),
) -> None:
    """Assign stable splits to grounded-QA raw examples."""

    metrics = split_examples(
        in_path=in_path,
        out_path=out,
        metrics_out=metrics_out,
        split_cfg=SplitConfig(
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            train_sft_ratio=train_sft_ratio,
            train_dpo_ratio=train_dpo_ratio,
        ),
    )
    console.print(f"Saved split-assigned rows to {out}")
    console.print_json(json.dumps(metrics))


@teacher_app.command("generate")
def teacher_generate(
    in_path: str = typer.Option(..., help="Input raw example JSONL path."),
    out: str = typer.Option(..., help="Output teacher candidate JSONL path."),
    prompt_style: str = typer.Option(TeacherGenerationConfig.prompt_style, help="Teacher prompt style name."),
    num_candidates_per_example: int = typer.Option(TeacherGenerationConfig.num_candidates_per_example, help="Number of candidates per example."),
    api_model_name: str = typer.Option(TeacherGenerationConfig.api_model_name, help="OpenAI-compatible teacher model id."),
    api_base_url: str = typer.Option(TeacherGenerationConfig.api_base_url, help="OpenAI-compatible API base URL."),
    api_key_env: str = typer.Option(TeacherGenerationConfig.api_key_env, help="Environment variable holding the API key."),
    api_timeout_seconds: float = typer.Option(TeacherGenerationConfig.api_timeout_seconds, help="Request timeout in seconds."),
    api_max_concurrency: int = typer.Option(TeacherGenerationConfig.api_max_concurrency, help="Maximum concurrent API requests."),
    api_max_retries: int = typer.Option(TeacherGenerationConfig.api_max_retries, help="Retry count for transient API errors."),
    api_backoff_base_seconds: float = typer.Option(TeacherGenerationConfig.api_backoff_base_seconds, help="Base retry backoff in seconds."),
    api_backoff_max_seconds: float = typer.Option(TeacherGenerationConfig.api_backoff_max_seconds, help="Max retry backoff in seconds."),
    max_new_tokens: int = typer.Option(TeacherGenerationConfig.max_new_tokens, help="Maximum completion tokens."),
    temperature: float = typer.Option(TeacherGenerationConfig.temperature, help="Sampling temperature."),
    top_p: float = typer.Option(TeacherGenerationConfig.top_p, help="Sampling top-p."),
    seed: int = typer.Option(TeacherGenerationConfig.seed, help="Random seed."),
    prefilter_tokenizer_name: str = typer.Option(TeacherGenerationConfig.prefilter_tokenizer_name, help="Tokenizer name used for prompt prefilter."),
    max_prompt_tokens: int = typer.Option(TeacherGenerationConfig.max_prompt_tokens, help="Maximum prompt tokens allowed before teacher generation."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional metrics JSON path."),
) -> None:
    """Generate teacher candidates with an OpenAI-compatible API."""

    metrics = generate_teacher_candidates(
        in_path=in_path,
        out_path=out,
        metrics_out=metrics_out,
        cfg=TeacherGenerationConfig(
            prompt_style=prompt_style,
            num_candidates_per_example=num_candidates_per_example,
            api_model_name=api_model_name,
            api_base_url=api_base_url,
            api_key_env=api_key_env,
            api_timeout_seconds=api_timeout_seconds,
            api_max_concurrency=api_max_concurrency,
            api_max_retries=api_max_retries,
            api_backoff_base_seconds=api_backoff_base_seconds,
            api_backoff_max_seconds=api_backoff_max_seconds,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            prefilter_tokenizer_name=prefilter_tokenizer_name,
            max_prompt_tokens=max_prompt_tokens,
        ),
    )
    console.print(f"Saved teacher candidates to {out}")
    console.print_json(json.dumps(metrics))


@teacher_app.command("validate")
def teacher_validate(
    in_path: str = typer.Option(..., help="Input teacher candidate JSONL path."),
    out: str = typer.Option(..., help="Output validated candidate JSONL path."),
    selected_out: Optional[str] = typer.Option(None, help="Optional output path for one selected candidate per example."),
    enable_semantics: bool = typer.Option(ValidationConfig.enable_semantics, help="Enable structured semantic verification."),
    semantic_drop_by_nli: bool = typer.Option(ValidationConfig.semantic_drop_by_nli, help="Drop samples rejected by the semantic verifier."),
    semantic_decision_source: str = typer.Option(ValidationConfig.semantic_decision_source, help="Semantic decision source: full_binary or reject_aware."),
    tokenizer_name: str = typer.Option(ValidationConfig.tokenizer_name, help="Tokenizer name used for completion length checks."),
    max_completion_tokens: int = typer.Option(ValidationConfig.max_completion_tokens, help="Maximum canonical completion tokens."),
    semantic_match_f1_threshold: float = typer.Option(CorrectnessConfig.semantic_match_f1_threshold, help="Reference-answer F1 threshold."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional metrics JSON path."),
) -> None:
    """Parse, validate, and score teacher candidates."""

    metrics = validate_teacher_candidates(
        in_path=in_path,
        out_path=out,
        selected_out_path=selected_out,
        metrics_out=metrics_out,
        validation_cfg=ValidationConfig(
            enable_semantics=enable_semantics,
            semantic_drop_by_nli=semantic_drop_by_nli,
            semantic_decision_source=semantic_decision_source,
            tokenizer_name=tokenizer_name,
            max_completion_tokens=max_completion_tokens,
        ),
        correctness_cfg=CorrectnessConfig(
            semantic_match_f1_threshold=semantic_match_f1_threshold,
        ),
    )
    console.print(f"Saved validated candidates to {out}")
    if selected_out:
        console.print(f"Saved selected candidates to {selected_out}")
    console.print_json(json.dumps(metrics))


@teacher_app.command("validate-unanswerable")
def teacher_validate_unanswerable(
    in_path: str = typer.Option(..., help="Input unanswerable teacher candidate JSONL path."),
    out: str = typer.Option(..., help="Output validated unanswerable candidate JSONL path."),
    selected_out: Optional[str] = typer.Option(None, help="Optional output path for one selected candidate per example."),
    tokenizer_name: str = typer.Option(ValidationConfig.tokenizer_name, help="Tokenizer name used for completion length checks."),
    max_completion_tokens: int = typer.Option(ValidationConfig.max_completion_tokens, help="Maximum canonical completion tokens."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional metrics JSON path."),
) -> None:
    """Parse and validate unanswerable teacher candidates with unanswerable-specific rules."""

    metrics = validate_unanswerable_teacher_candidates(
        in_path=in_path,
        out_path=out,
        selected_out_path=selected_out,
        metrics_out=metrics_out,
        validation_cfg=ValidationConfig(
            enable_semantics=False,
            semantic_drop_by_nli=False,
            semantic_decision_source="full_binary",
            tokenizer_name=tokenizer_name,
            max_completion_tokens=max_completion_tokens,
        ),
    )
    console.print(f"Saved validated unanswerable candidates to {out}")
    if selected_out:
        console.print(f"Saved selected unanswerable candidates to {selected_out}")
    console.print_json(json.dumps(metrics))


@build_app.command("sft-records")
def build_sft(
    in_path: str = typer.Option(..., help="Input selected validated JSONL path."),
    out: str = typer.Option(..., help="Output SFT records JSONL path."),
    prompt_style: str = typer.Option(SftRecordConfig.prompt_style, help="Inference prompt style for the student."),
    keep_only_overall_ok: bool = typer.Option(SftRecordConfig.keep_only_overall_ok, help="Keep only overall-valid candidates."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional metrics JSON path."),
) -> None:
    """Build SFT prompt-completion records from validated candidates."""

    metrics = build_sft_records(
        in_path=in_path,
        out_path=out,
        metrics_out=metrics_out,
        prompt_style=prompt_style,
        keep_only_overall_ok=keep_only_overall_ok,
    )
    console.print(f"Saved SFT records to {out}")
    console.print_json(json.dumps(metrics))


if __name__ == "__main__":
    app()
