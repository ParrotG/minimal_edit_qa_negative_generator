from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HaluEvalSourceConfig:
    """Configuration for loading QA source records from HaluEval."""

    dataset_name: str = "pminervini/HaluEval"
    subset: str = "qa"
    split: str = "data"
    knowledge_field: str = "knowledge"
    question_field: str = "question"
    answer_field: str = "right_answer"


@dataclass(frozen=True)
class SourceLengthConfig:
    """Length constraints for source-record filtering."""

    tokenizer_name: str = "Qwen/Qwen3-0.6B"
    max_prompt_tokens: int = 768
    max_answer_tokens: int = 256
    max_total_tokens: int = 1024


@dataclass(frozen=True)
class GenerationConfig:
    """Generation configuration for self-sampling."""

    backend: str = "local"  # local | api
    model_name: str = "Qwen/Qwen3-0.6B"
    device: str = "cuda"
    batch_size: int = 4
    max_new_tokens: int = 512
    max_answer_tokens: int = 256
    max_attempts_per_sample: int = 3
    num_samples_per_question: int = 6
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    use_chat_template: bool = False
    enable_thinking: bool = False
    strip_think_tags: bool = True
    strip_role_markers: bool = True
    strip_markdown_fences: bool = True
    extract_short_answer: bool = True
    include_generation_meta: bool = False

    # API backend settings (OpenAI-compatible endpoint).
    api_model_name: str = "qwen3-0.6b"
    api_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    api_key_env: str = "DASHSCOPE_API_KEY"
    api_timeout_seconds: float = 90.0
    api_max_concurrency: int = 64
    api_max_retries: int = 4
    api_retry_backoff_base: float = 0.5
    api_retry_backoff_max: float = 8.0
    seed: int = 42


@dataclass(frozen=True)
class NLIConfig:
    """NLI verifier configuration."""

    model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    device: str = "cuda"
    batch_size: int = 16
    max_length: int = 512
    fp16: bool = True


@dataclass(frozen=True)
class JudgeConfig:
    """Judgment thresholds and optional semantic consistency constraints."""

    reference_entail_threshold: float = 0.60
    candidate_entail_threshold: float = 0.45
    candidate_contradict_threshold: float = 0.90
    vote_mode: str = "primary"  # primary | and | or
    qa_similarity_model_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"
    qa_similarity_min: float = 0.65


@dataclass(frozen=True)
class PairBuildConfig:
    """Configuration for converting judged answers into pair records."""

    min_trials_for_very_label: int = 10
    keep_easy_without_negative: bool = False


@dataclass(frozen=True)
class RepairConfig:
    """Configuration for minimal-edit repair on hard samples."""

    model_name: str = "Qwen/Qwen3-4B-Instruct"
    device: str = "cuda"
    batch_size: int = 2
    max_new_tokens: int = 256
    attempts_per_record: int = 4
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0
    entail_threshold: float = 0.60
    min_norm_edit: float = 0.01
    max_norm_edit: float = 0.35
    min_length_ratio: float = 0.75
    max_length_ratio: float = 1.35
    fallback_to_reference: bool = False
    seed: int = 42


@dataclass(frozen=True)
class PairFilterConfig:
    """Filtering configuration for final DPO pair export."""

    min_norm_edit: float = 0.01
    max_norm_edit: float = 0.40
    min_length_ratio: float = 0.75
    max_length_ratio: float = 1.35
    enforce_answer_type: bool = True
    spacy_model: str = "en_core_web_trf"
    qa_similarity_model_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"
    qa_similarity_min: float = 0.60


@dataclass(frozen=True)
class AuditConfig:
    """Configuration for simple surface-signal audit."""

    separability_threshold: float = 0.65


@dataclass(frozen=True)
class DifficultyConfig:
    """Configuration for policy-logprob difficulty bucketing."""

    model_name: str = "Qwen/Qwen3-0.6B"
    device: str = "cuda"
    batch_size: int = 2
    max_length: int = 2048
    fp16: bool = True
    hard_max_delta: float = 0.10
    medium_max_delta: float = 0.60
