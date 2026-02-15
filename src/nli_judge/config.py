from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NLIConfig:
    """Default runtime configuration for NLI verification."""

    model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    device: str = "cuda"
    batch_size: int = 16
    max_length: int = 512
    fp16: bool = True
