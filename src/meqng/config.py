from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HaluEvalConfig:
    """Configuration for loading HaluEval from Hugging Face Datasets."""
    dataset_name: str = "pminervini/HaluEval"
    # The upstream dataset includes multiple configs (qa/dialogue/summarization).
    # We default to "qa".
    subset: str = "qa"
    split: str = "data"
    # Field names (qa config is expected to have both right and hallucinated answers).
    knowledge_field: str = "knowledge"
    question_field: str = "question"
    right_answer_field: str = "right_answer"
    hallucinated_answer_field: str = "hallucinated_answer"


@dataclass(frozen=True)
class NLIConfig:
    """Configuration for NLI verifier."""
    model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    batch_size: int = 16
    max_length: int = 512
    device: str = "cuda"  # "cuda" or "cpu"
    fp16: bool = True
    entail_threshold_pos: float = 0.60
    entail_threshold_neg: float = 0.30
