from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console

from qa_checks.correctness import CorrectnessConfig
from qa_data import ConstructionConfig, HotpotSourceConfig, NegativeSamplingConfig, SplitConfig

from .config import SftRecordConfig, TeacherGenerationConfig, ValidationConfig
from .workflow import (
    build_hotpot_source,
    build_sft_records,
    build_simple_negatives,
    generate_teacher_candidates,
    split_examples,
    validate_teacher_candidates,
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
    """Derive simple first-pass unanswerable examples."""

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
    use_support_window_knowledge: bool = typer.Option(ValidationConfig.use_support_window_knowledge, help="Prefer support-window knowledge for semantics."),
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
            use_support_window_knowledge=use_support_window_knowledge,
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
