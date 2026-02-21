from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class NLIConfig:
    """Default runtime configuration for NLI verification."""

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
