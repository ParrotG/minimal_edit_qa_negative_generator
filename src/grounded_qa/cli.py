from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console

from project_config import PROJECT_SETTINGS
from qa_checks.correctness import CorrectnessConfig
from qa_data import (
    AnswerabilitySplitConfig,
    ConstructionConfig,
    DataSplitConfig,
    HotpotSourceConfig,
    UnanswerableBuildConfig,
)

from .config import SftRecordConfig, SourcePrefilterConfig, TeacherGenerationConfig, ValidationConfig
from .workflow import (
    build_sft_records,
    build_source_examples,
    generate_teacher_candidates,
    partition_source_examples,
    prefilter_source_examples,
    tag_source_rows,
    validate_mixed_teacher_candidates,
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

    console.print("grounded_qa 0.2.0")


@source_app.command("tag")
def source_tag(
    out: str = typer.Option(..., help="Output tagged Hotpot JSONL path."),
    dataset_name: str = typer.Option(PROJECT_SETTINGS.source.dataset_name, help="HotpotQA dataset name."),
    train_split: str = typer.Option(PROJECT_SETTINGS.source.train_split, help="HotpotQA train split used for SFT training data."),
    validation_split: str = typer.Option(PROJECT_SETTINGS.source.validation_split, help="HotpotQA validation split used for held-out validation/test data."),
    max_train_samples: int = typer.Option(PROJECT_SETTINGS.source.max_train_samples, help="Maximum source rows to load from the Hotpot train split. -1 keeps all."),
    max_validation_samples: int = typer.Option(PROJECT_SETTINGS.source.max_validation_samples, help="Maximum source rows to load from the Hotpot validation split. -1 keeps all."),
    seed: int = typer.Option(PROJECT_SETTINGS.source.seed, help="Random seed."),
    validation_ratio: float = typer.Option(PROJECT_SETTINGS.source.validation_ratio, help="Validation ratio applied inside the Hotpot validation subset."),
    test_ratio: float = typer.Option(PROJECT_SETTINGS.source.test_ratio, help="Test ratio applied inside the Hotpot validation subset."),
    answerable_ratio: float = typer.Option(PROJECT_SETTINGS.source.answerable_ratio, help="Answerable construction ratio."),
    unanswerable_ratio: float = typer.Option(PROJECT_SETTINGS.source.unanswerable_ratio, help="Unanswerable construction ratio."),
    both_ratio: float = typer.Option(PROJECT_SETTINGS.source.both_ratio, help="Both-track construction ratio."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional metrics JSON path."),
) -> None:
    """Assign independent data and answerability splits to Hotpot rows."""

    metrics = tag_source_rows(
        out_path=out,
        metrics_out=metrics_out,
        source_cfg=HotpotSourceConfig(
            dataset_name=dataset_name,
            train_split=train_split,
            validation_split=validation_split,
            max_train_samples=max_train_samples,
            max_validation_samples=max_validation_samples,
            seed=seed,
        ),
        data_split_cfg=DataSplitConfig(
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        ),
        answerability_split_cfg=AnswerabilitySplitConfig(
            answerable_ratio=answerable_ratio,
            unanswerable_ratio=unanswerable_ratio,
            both_ratio=both_ratio,
        ),
    )
    console.print(f"Saved tagged Hotpot rows to {out}")
    console.print_json(json.dumps(metrics))


@source_app.command("build")
def source_build(
    in_path: str = typer.Option(..., help="Input tagged Hotpot JSONL path."),
    out: str = typer.Option(..., help="Output mixed prepared examples JSONL path."),
    window_size: int = typer.Option(PROJECT_SETTINGS.source.window_size, help="Support window size."),
    max_supporting_facts: int = typer.Option(PROJECT_SETTINGS.source.max_supporting_facts, help="Maximum supporting facts."),
    include_title_prefix: bool = typer.Option(PROJECT_SETTINGS.source.include_title_prefix, help="Whether to prefix knowledge blocks with titles."),
    replace_supporting_facts_min: int = typer.Option(PROJECT_SETTINGS.source.replace_supporting_facts_min, help="Minimum number of supporting facts to replace."),
    replace_supporting_facts_max: int = typer.Option(PROJECT_SETTINGS.source.replace_supporting_facts_max, help="Maximum number of supporting facts to replace."),
    same_doc_candidate_radius: int = typer.Option(PROJECT_SETTINGS.source.same_doc_candidate_radius, help="Same-document neighbor radius for replacement sentences."),
    allow_same_doc_non_adjacent: bool = typer.Option(PROJECT_SETTINGS.source.allow_same_doc_non_adjacent, help="Allow fallback to same-document non-adjacent sentences."),
    adjacent_doc_sentence_limit: int = typer.Option(PROJECT_SETTINGS.source.adjacent_doc_sentence_limit, help="Maximum number of sentences to consider from adjacent documents."),
    seed: int = typer.Option(PROJECT_SETTINGS.source.seed, help="Random seed."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional metrics JSON path."),
) -> None:
    """Build mixed answerable and unanswerable concrete examples from tagged rows."""

    metrics = build_source_examples(
        in_path=in_path,
        out_path=out,
        metrics_out=metrics_out,
        construct_cfg=ConstructionConfig(
            window_size=window_size,
            max_supporting_facts=max_supporting_facts,
            include_title_prefix=include_title_prefix,
        ),
        unanswerable_cfg=UnanswerableBuildConfig(
            replace_supporting_facts_min=replace_supporting_facts_min,
            replace_supporting_facts_max=replace_supporting_facts_max,
            same_doc_candidate_radius=same_doc_candidate_radius,
            allow_same_doc_non_adjacent=allow_same_doc_non_adjacent,
            adjacent_doc_sentence_limit=adjacent_doc_sentence_limit,
            include_title_prefix=include_title_prefix,
            seed=seed,
        ),
    )
    console.print(f"Saved prepared examples to {out}")
    console.print_json(json.dumps(metrics))


@source_app.command("prefilter")
def source_prefilter(
    in_path: str = typer.Option(..., help="Input prepared examples JSONL path."),
    out: str = typer.Option(..., help="Output source-prefiltered JSONL path."),
    tokenizer_name: str = typer.Option(PROJECT_SETTINGS.model.default_tokenizer_name, help="Tokenizer name used for infer-prompt budgeting."),
    max_prompt_tokens: int = typer.Option(PROJECT_SETTINGS.token_budget.prompt_limit, help="Maximum infer prompt tokens."),
    enable_unanswerable_nli: bool = typer.Option(PROJECT_SETTINGS.source.enable_unanswerable_nli, help="Enable NLI prefiltering for unanswerable examples."),
    nli_model_name: str = typer.Option(PROJECT_SETTINGS.nli.model_name, help="NLI model name."),
    nli_device: str = typer.Option(PROJECT_SETTINGS.nli.device, help="NLI device."),
    nli_batch_size: int = typer.Option(PROJECT_SETTINGS.nli.batch_size, help="NLI batch size."),
    nli_max_length: int = typer.Option(PROJECT_SETTINGS.nli.max_length, help="NLI max length."),
    nli_fp16: bool = typer.Option(PROJECT_SETTINGS.nli.fp16, help="Whether to use fp16 for the NLI verifier."),
    temperature: float = typer.Option(PROJECT_SETTINGS.judge.temperature, help="Temperature used by the calibrated NLI judge."),
    full_margin_threshold: float = typer.Option(PROJECT_SETTINGS.judge.full_margin_threshold, help="Full-binary margin threshold."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional metrics JSON path."),
) -> None:
    """Apply mixed source-stage token and unanswerable NLI filtering."""

    metrics = prefilter_source_examples(
        in_path=in_path,
        out_path=out,
        metrics_out=metrics_out,
        cfg=SourcePrefilterConfig(
            tokenizer_name=tokenizer_name,
            max_prompt_tokens=max_prompt_tokens,
            enable_unanswerable_nli=enable_unanswerable_nli,
            nli_model_name=nli_model_name,
            nli_device=nli_device,
            nli_batch_size=nli_batch_size,
            nli_max_length=nli_max_length,
            nli_fp16=nli_fp16,
            temperature=temperature,
            full_margin_threshold=full_margin_threshold,
        ),
    )
    console.print(f"Saved source-prefiltered examples to {out}")
    console.print_json(json.dumps(metrics))


@source_app.command("partition")
def source_partition(
    in_path: str = typer.Option(..., help="Input source-prefiltered JSONL path."),
    out_dir: str = typer.Option(..., help="Output directory containing partitioned JSONL files."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional metrics JSON path."),
) -> None:
    """Partition mixed examples by downstream data split."""

    metrics = partition_source_examples(
        in_path=in_path,
        out_dir=out_dir,
        metrics_out=metrics_out,
    )
    console.print(f"Saved partitioned examples to {out_dir}")
    console.print_json(json.dumps(metrics))


@teacher_app.command("generate")
def teacher_generate(
    in_path: str = typer.Option(..., help="Input mixed example JSONL path."),
    out: str = typer.Option(..., help="Output teacher candidate JSONL path."),
    prompt_style: str = typer.Option(PROJECT_SETTINGS.teacher.prompt_style, help="Teacher prompt style name."),
    answerable_num_candidates_per_example: int = typer.Option(
        PROJECT_SETTINGS.teacher.answerable_num_candidates_per_example,
        help="Number of candidates per answerable example.",
    ),
    unanswerable_num_candidates_per_example: int = typer.Option(
        PROJECT_SETTINGS.teacher.unanswerable_num_candidates_per_example,
        help="Number of candidates per unanswerable example.",
    ),
    api_model_name: str = typer.Option(PROJECT_SETTINGS.teacher_api.model_name, help="OpenAI-compatible teacher model id."),
    api_base_url: str = typer.Option(PROJECT_SETTINGS.teacher_api.base_url, help="OpenAI-compatible API base URL."),
    api_key_env: str = typer.Option(PROJECT_SETTINGS.teacher_api.api_key_env, help="Environment variable holding the API key."),
    api_timeout_seconds: float = typer.Option(PROJECT_SETTINGS.teacher_api.timeout_seconds, help="Request timeout in seconds."),
    api_max_concurrency: int = typer.Option(PROJECT_SETTINGS.teacher_api.max_concurrency, help="Maximum concurrent API requests."),
    api_max_retries: int = typer.Option(PROJECT_SETTINGS.teacher_api.max_retries, help="Retry count for transient API errors."),
    api_backoff_base_seconds: float = typer.Option(PROJECT_SETTINGS.teacher_api.backoff_base_seconds, help="Base retry backoff in seconds."),
    api_backoff_max_seconds: float = typer.Option(PROJECT_SETTINGS.teacher_api.backoff_max_seconds, help="Max retry backoff in seconds."),
    max_new_tokens: int = typer.Option(PROJECT_SETTINGS.teacher_api.max_tokens, help="Maximum completion tokens."),
    temperature: float = typer.Option(PROJECT_SETTINGS.teacher_api.temperature, help="Sampling temperature."),
    top_p: float = typer.Option(PROJECT_SETTINGS.teacher_api.top_p, help="Sampling top-p."),
    seed: int = typer.Option(PROJECT_SETTINGS.teacher_api.seed, help="Random seed."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional metrics JSON path."),
) -> None:
    """Generate teacher candidates with an OpenAI-compatible API."""

    metrics = generate_teacher_candidates(
        in_path=in_path,
        out_path=out,
        metrics_out=metrics_out,
        cfg=TeacherGenerationConfig(
            prompt_style=prompt_style,
            answerable_num_candidates_per_example=answerable_num_candidates_per_example,
            unanswerable_num_candidates_per_example=unanswerable_num_candidates_per_example,
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
        ),
    )
    console.print(f"Saved teacher candidates to {out}")
    console.print_json(json.dumps(metrics))


@teacher_app.command("validate")
def teacher_validate(
    in_path: str = typer.Option(..., help="Input mixed teacher candidate JSONL path."),
    out: str = typer.Option(..., help="Output validated candidate JSONL path."),
    selected_out: Optional[str] = typer.Option(None, help="Optional output path for one selected candidate per example."),
    enable_semantics: bool = typer.Option(PROJECT_SETTINGS.teacher.validate_enable_semantics, help="Enable structured semantic verification for answerable candidates."),
    semantic_drop_by_nli: bool = typer.Option(PROJECT_SETTINGS.teacher.semantic_drop_by_nli, help="Drop answerable samples rejected by the semantic verifier."),
    semantic_decision_source: str = typer.Option(PROJECT_SETTINGS.teacher.semantic_decision_source, help="Semantic decision source: full_binary or reject_aware."),
    tokenizer_name: str = typer.Option(PROJECT_SETTINGS.model.default_tokenizer_name, help="Tokenizer name used for completion length checks."),
    max_completion_tokens: int = typer.Option(PROJECT_SETTINGS.token_budget.completion_limit, help="Maximum canonical completion tokens."),
    semantic_match_f1_threshold: float = typer.Option(PROJECT_SETTINGS.correctness.semantic_match_f1_threshold, help="Reference-answer F1 threshold."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional metrics JSON path."),
) -> None:
    """Parse, validate, and select mixed teacher candidates."""

    metrics = validate_mixed_teacher_candidates(
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


@build_app.command("sft-records")
def build_sft_records_cmd(
    in_path: str = typer.Option(..., help="Input selected validated JSONL path."),
    out: str = typer.Option(..., help="Output SFT record JSONL path."),
    prompt_style: str = typer.Option(PROJECT_SETTINGS.teacher.infer_prompt_style, help="SFT prompt style."),
    keep_only_overall_ok: bool = typer.Option(PROJECT_SETTINGS.teacher.keep_only_overall_ok, help="Keep only rows whose validation report is overall_ok."),
    metrics_out: Optional[str] = typer.Option(None, help="Optional metrics JSON path."),
) -> None:
    """Build SFT prompt-completion records from mixed selected rows."""

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
