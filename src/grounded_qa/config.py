from __future__ import annotations

from dataclasses import dataclass

from llm_textgen.api_client import ApiGenerationConfig
from project_config import PROJECT_SETTINGS
from qa_judge.config import JudgeConfig, NLIConfig


@dataclass(frozen=True)
class TeacherGenerationConfig:
    """Defaults for teacher candidate generation."""

    prompt_style: str = "teacher_v1"
    answerable_num_candidates_per_example: int = 3
    unanswerable_num_candidates_per_example: int = 1
    api_model_name: str = PROJECT_SETTINGS.teacher_api.model_name
    api_base_url: str = PROJECT_SETTINGS.teacher_api.base_url
    api_key_env: str = PROJECT_SETTINGS.teacher_api.api_key_env
    api_timeout_seconds: float = PROJECT_SETTINGS.teacher_api.timeout_seconds
    api_max_concurrency: int = PROJECT_SETTINGS.teacher_api.max_concurrency
    api_max_retries: int = PROJECT_SETTINGS.teacher_api.max_retries
    api_backoff_base_seconds: float = PROJECT_SETTINGS.teacher_api.backoff_base_seconds
    api_backoff_max_seconds: float = PROJECT_SETTINGS.teacher_api.backoff_max_seconds
    max_new_tokens: int = PROJECT_SETTINGS.teacher_api.max_tokens
    temperature: float = PROJECT_SETTINGS.teacher_api.temperature
    top_p: float = PROJECT_SETTINGS.teacher_api.top_p
    seed: int = PROJECT_SETTINGS.teacher_api.seed
    error_log_dir: str = PROJECT_SETTINGS.paths.error_log_dir

    def to_api_config(self) -> ApiGenerationConfig:
        """Convert to the reusable API generation config."""

        return ApiGenerationConfig(
            model_name=self.api_model_name,
            base_url=self.api_base_url,
            timeout_seconds=self.api_timeout_seconds,
            max_concurrency=self.api_max_concurrency,
            max_retries=self.api_max_retries,
            backoff_base_seconds=self.api_backoff_base_seconds,
            backoff_max_seconds=self.api_backoff_max_seconds,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            error_log_dir=self.error_log_dir,
        )


@dataclass(frozen=True)
class SourcePrefilterConfig:
    """Source-stage filtering defaults for mixed prepared examples."""

    tokenizer_name: str = PROJECT_SETTINGS.model.default_tokenizer_name
    max_prompt_tokens: int = PROJECT_SETTINGS.protocol.max_prompt_tokens
    enable_unanswerable_nli: bool = True
    nli_model_name: str = NLIConfig.model_name
    nli_device: str = NLIConfig.device
    nli_batch_size: int = NLIConfig.batch_size
    nli_max_length: int = NLIConfig.max_length
    nli_fp16: bool = NLIConfig.fp16
    temperature: float = JudgeConfig.temperature
    full_margin_threshold: float = JudgeConfig.full_margin_threshold


@dataclass(frozen=True)
class ValidationConfig:
    """Validation defaults for teacher candidates."""

    enable_semantics: bool = True
    semantic_drop_by_nli: bool = True
    semantic_decision_source: str = "full_binary"
    tokenizer_name: str = PROJECT_SETTINGS.model.default_tokenizer_name
    max_completion_tokens: int = PROJECT_SETTINGS.protocol.max_completion_tokens


@dataclass(frozen=True)
class SftRecordConfig:
    """Configuration for SFT record packing."""

    prompt_style: str = "infer_v1"
    keep_only_overall_ok: bool = True
