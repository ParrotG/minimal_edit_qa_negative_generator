from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ProtocolSpec:
    """Configuration for one structured grounded-QA protocol version."""

    version: str = "grounded-qa-v1"
    max_evidence_count: int = 4
    min_quote_chars: int = 8
    max_quote_chars: int = 220
    min_rationale_chars: int = 8
    max_rationale_chars: int = 320
    max_answer_chars: int = 256
    max_total_chars: int = 4000
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
