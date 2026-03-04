from __future__ import annotations

from dataclasses import dataclass

from llm_textgen.api_client import ApiGenerationConfig
from qa_judge.config import JudgeConfig, NLIConfig


@dataclass(frozen=True)
class TeacherGenerationConfig:
    """Defaults for teacher candidate generation."""

    prompt_style: str = "teacher_v1"
    answerable_num_candidates_per_example: int = 3
    unanswerable_num_candidates_per_example: int = 1
    api_model_name: str = "qwen3.5-plus"
    api_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    api_key_env: str = "DASHSCOPE_API_KEY"
    api_timeout_seconds: float = 90.0
    api_max_concurrency: int = 8
    api_max_retries: int = 4
    api_backoff_base_seconds: float = 0.5
    api_backoff_max_seconds: float = 8.0
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    seed: int = 42
    error_log_dir: str = "log"

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

    tokenizer_name: str = "Qwen/Qwen3-0.6B"
    max_prompt_tokens: int = 512
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
    tokenizer_name: str = "Qwen/Qwen3-0.6B"
    max_completion_tokens: int = 512


@dataclass(frozen=True)
class SftRecordConfig:
    """Configuration for SFT record packing."""

    prompt_style: str = "infer_v1"
    keep_only_overall_ok: bool = True
