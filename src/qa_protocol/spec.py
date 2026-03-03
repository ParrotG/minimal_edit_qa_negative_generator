from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ProtocolSpec:
    """Configuration for one structured grounded-QA protocol version."""

    version: str = "grounded-qa-v1"
    max_evidence_count: int = 4
    validation_tokenizer_name: str = "Qwen/Qwen3-0.6B"
    max_completion_tokens: int = 512
    prefilter_tokenizer_name: str = "Qwen/Qwen3-0.6B"
    max_prompt_tokens: int = 512
    canonical_refusal: str = "I don't know based on the provided knowledge."
    refusal_templates: Tuple[str, ...] = (
        "I don't know based on the provided knowledge.",
        "The provided knowledge does not contain enough information to answer the question.",
        "I cannot answer from the provided knowledge.",
    )
    field_order: Tuple[str, ...] = (
        "answerability",
        "evidence",
        "rationale",
        "answer",
        "confidence",
    )


DEFAULT_PROTOCOL_SPEC = ProtocolSpec()
