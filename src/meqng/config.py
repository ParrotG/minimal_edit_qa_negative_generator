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


@dataclass(frozen=True)
class SpanDropConfig:
    """Configuration for lightweight span-level perturbation."""
    min_words: int = 2
    max_words: int = 6


@dataclass(frozen=True)
class TextAttackConfig:
    """Configuration for TextAttack-based NLI flip perturbation."""
    augmenter: str = "embedding"
    pct_words_to_swap: float = 0.15
    transformations_per_example: int = 8
    search_calls: int = 6
    entail_threshold_neg: float = 0.35
    contradiction_ratio: float = 0.50
    min_norm_edit: float = 0.01
    max_norm_edit: float = 0.25
    semantic_model_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"
    min_semantic_similarity: float = 0.80
    protect_entities: bool = True
    protect_numbers: bool = True
    spacy_model: str = "en_core_web_trf"


@dataclass(frozen=True)
class FilterConfig:
    """Configuration for filtering and ranking stage."""

    qa_similarity_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    qa_similarity_min: float = 0.70
    grammar_max_extra_issues: int = 1


@dataclass(frozen=True)
class DifficultyConfig:
    """Configuration for difficulty scoring and bucketing."""

    model_name: str = "Qwen/Qwen3-0.6B"
    batch_size: int = 2
    max_length: int = 2048
    device: str = "cuda"
    fp16: bool = True
    hard_max_delta: float = 0.10
    medium_max_delta: float = 0.60
