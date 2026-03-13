from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ModelSettings:
    """Project-wide defaults for the target training model family."""

    target_training_llm: str = "Qwen/Qwen3-0.6B"
    default_tokenizer_name: str = "Qwen/Qwen3-0.6B"


@dataclass(frozen=True)
class TeacherApiSettings:
    """Default API settings for teacher data generation."""

    model_name: str = "qwen3.5-plus"
    base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    api_key_env: str = "DASHSCOPE_API_KEY"
    timeout_seconds: float = 90.0
    max_concurrency: int = 8
    max_retries: int = 4
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 8.0
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    seed: int = 42


@dataclass(frozen=True)
class AnswerExtractionApiSettings:
    """Default API settings for answer extraction and refusal detection."""

    model_name: str = "qwen-plus"
    base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    api_key_env: str = "DASHSCOPE_API_KEY"
    timeout_seconds: float = 90.0
    max_concurrency: int = 64
    max_retries: int = 4
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 8.0
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.95
    seed: int = 42
    max_attempts: int = 2


@dataclass(frozen=True)
class NliSettings:
    """Default runtime settings for NLI verification."""

    model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    device: str = "cuda"
    batch_size: int = 16
    max_length: int = 512
    fp16: bool = True


@dataclass(frozen=True)
class JudgeSettings:
    """Default calibrated decision settings for answer judgment."""

    temperature: float = 3.0
    full_margin_threshold: float = 0.6
    reject_margin_threshold: float = 0.9
    reject_band_half_width: float = 0.95
    qa_fail_as_negative: bool = True
    qa_check_answer_type: bool = True
    qa_spacy_model: str = "en_core_web_trf"


@dataclass(frozen=True)
class ProtocolSettings:
    """Protocol defaults shared by prompting and validation."""

    version: str = "grounded-qa-v1"
    max_evidence_count: int = 4
    max_completion_tokens: int = 512
    max_prompt_tokens: int = 512
    canonical_refusal: str = "I don't know based on the provided knowledge."
    refusal_templates: Tuple[str, ...] = (
        "I don't know based on the provided knowledge.",
        "The provided knowledge does not contain enough information to answer the question.",
        "I cannot answer from the provided knowledge.",
    )
    field_order: Tuple[str, ...] = (
        "answerability",
        "evidence",
        "rationale",
        "answer",
        "confidence",
    )


@dataclass(frozen=True)
class TokenBudgetSettings:
    """Defaults for tokenizer-based budgeting and token accounting."""

    count_batch_size: int = 128
    record_token_usage: bool = False


@dataclass(frozen=True)
class GenerationSettings:
    """Local generation defaults."""

    batch_size: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    use_chat_template: bool = True
    enable_thinking: bool = False
    strip_think_tags: bool = True
    strip_role_markers: bool = True
    device: str = "cuda"
    device_map: str | None = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False
    padding_side: str = "left"
    use_fast_tokenizer: bool = True
    seed: int = 42


@dataclass(frozen=True)
class CalibrationSettings:
    """Defaults for calibration data preparation and search."""

    group_by: str = "task"
    search_objective: str = "f1"
    reject_alpha: float = 0.25
    positive_label_values: str = "1,true,yes,supported,faithful,correct,entail"
    negative_label_values: str = "0,false,no,unsupported,unfaithful,incorrect,neutral,contradict"
    margin_threshold_min: float = -4.0
    margin_threshold_max: float = 4.0
    margin_threshold_step: float = 0.1
    argmax_conf_threshold_min: float = 0.34
    argmax_conf_threshold_max: float = 0.99
    argmax_conf_threshold_step: float = 0.01
    band_half_width_min: float = 0.0
    band_half_width_max: float = 1.0
    band_half_width_step: float = 0.05
    matcher_threshold_min: float = 0.0
    matcher_threshold_max: float = 1.0
    matcher_threshold_step: float = 0.01
    matcher_runtime_threshold: float = 0.5
    matcher_model_name: str = "zli12321/answer_equivalence_roberta-large"
    annotation_task_types: Tuple[str, ...] = ("nli_flat", "nli_structured", "matcher")


@dataclass(frozen=True)
class PathSettings:
    """Common filesystem defaults."""

    error_log_dir: str = "log"


@dataclass(frozen=True)
class ProjectSettings:
    """Single source of truth for project defaults."""

    model: ModelSettings = ModelSettings()
    teacher_api: TeacherApiSettings = TeacherApiSettings()
    answer_extraction_api: AnswerExtractionApiSettings = AnswerExtractionApiSettings()
    nli: NliSettings = NliSettings()
    judge: JudgeSettings = JudgeSettings()
    protocol: ProtocolSettings = ProtocolSettings()
    token_budget: TokenBudgetSettings = TokenBudgetSettings()
    generation: GenerationSettings = GenerationSettings()
    calibration: CalibrationSettings = CalibrationSettings()
    paths: PathSettings = PathSettings()


PROJECT_SETTINGS = ProjectSettings()
