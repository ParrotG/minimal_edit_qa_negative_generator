from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from project_config import PROJECT_SETTINGS


@dataclass(frozen=True)
class UnifiedLLMConfig:
    """Configuration for unified local and LoRA text generation."""

    # Model loading defaults.
    base_model_name: str = PROJECT_SETTINGS.model.target_training_llm
    lora_path: Optional[str] = None
    tokenizer_name_or_path: Optional[str] = None
    device: str = PROJECT_SETTINGS.generation.device
    device_map: Optional[str] = PROJECT_SETTINGS.generation.device_map
    dtype: str = PROJECT_SETTINGS.generation.dtype
    trust_remote_code: bool = PROJECT_SETTINGS.generation.trust_remote_code
    padding_side: str = PROJECT_SETTINGS.generation.padding_side
    use_fast_tokenizer: bool = PROJECT_SETTINGS.generation.use_fast_tokenizer
    seed: int = PROJECT_SETTINGS.generation.seed

    # Generation defaults.
    batch_size: int = PROJECT_SETTINGS.generation.batch_size
    max_new_tokens: int = PROJECT_SETTINGS.generation.max_new_tokens
    temperature: float = PROJECT_SETTINGS.generation.temperature
    top_p: float = PROJECT_SETTINGS.generation.top_p
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: float = PROJECT_SETTINGS.generation.repetition_penalty
    use_chat_template: bool = PROJECT_SETTINGS.generation.use_chat_template
    enable_thinking: bool = PROJECT_SETTINGS.generation.enable_thinking
    strip_think_tags: bool = PROJECT_SETTINGS.generation.strip_think_tags
    strip_role_markers: bool = PROJECT_SETTINGS.generation.strip_role_markers
    record_token_usage: bool = PROJECT_SETTINGS.token_budget.record_token_usage
