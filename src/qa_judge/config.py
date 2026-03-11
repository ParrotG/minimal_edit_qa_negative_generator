from __future__ import annotations

from dataclasses import dataclass

from project_config import PROJECT_SETTINGS


@dataclass(frozen=True)
class NLIConfig:
    """Default runtime configuration for NLI verification."""

    model_name: str = PROJECT_SETTINGS.nli.model_name
    device: str = PROJECT_SETTINGS.nli.device
    batch_size: int = PROJECT_SETTINGS.nli.batch_size
    max_length: int = PROJECT_SETTINGS.nli.max_length
    fp16: bool = PROJECT_SETTINGS.nli.fp16


@dataclass(frozen=True)
class JudgeConfig:
    """Calibrated NLI and QA-check configuration for answer judgment."""

    temperature: float = PROJECT_SETTINGS.judge.temperature
    full_margin_threshold: float = PROJECT_SETTINGS.judge.full_margin_threshold
    reject_margin_threshold: float = PROJECT_SETTINGS.judge.reject_margin_threshold
    reject_band_half_width: float = PROJECT_SETTINGS.judge.reject_band_half_width

    # QA single-answer checks.
    qa_fail_as_negative: bool = PROJECT_SETTINGS.judge.qa_fail_as_negative
    qa_check_answer_type: bool = PROJECT_SETTINGS.judge.qa_check_answer_type
    qa_spacy_model: str = PROJECT_SETTINGS.judge.qa_spacy_model
