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


@dataclass(frozen=True)
class JudgeConfig:
    """Calibrated NLI and QA-check configuration for answer judgment."""

    # Temperature scaling + strategy defaults from:
    # outputs/test_run/nli_deepeval_split_search_calibrated_best.jsonl
    temperature: float = 3.0
    full_margin_threshold: float = 0.6
    reject_margin_threshold: float = 0.9
    reject_band_half_width: float = 0.95

    # QA single-answer checks.
    qa_fail_as_negative: bool = True
    qa_check_answer_type: bool = True
    qa_spacy_model: str = "en_core_web_trf"
