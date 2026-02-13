from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class UnifiedLLMConfig:
    """Configuration for unified local and LoRA text generation."""

    # Model loading defaults.
    base_model_name: str = "Qwen/Qwen3-0.6B"
    lora_path: Optional[str] = None
    lora_is_merged: bool = False
    merge_lora: bool = False
    tokenizer_name_or_path: Optional[str] = None
    device: str = "cuda"
    device_map: Optional[str] = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False
    padding_side: str = "left"
    use_fast_tokenizer: bool = True
    seed: int = 42

    # Generation defaults.
    batch_size: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    use_chat_template: bool = True
    enable_thinking: bool = False
    strip_think_tags: bool = True
    strip_role_markers: bool = True
