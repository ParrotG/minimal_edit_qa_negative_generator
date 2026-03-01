from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Answerability(StrEnum):
    """Answerability labels in the structured QA protocol."""

    ANSWERABLE = "answerable"
    UNANSWERABLE = "unanswerable"


class ConfidenceLevel(StrEnum):
    """Confidence labels in the structured QA protocol."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceQuote(BaseModel):
    """One evidence quote selected from the provided knowledge."""

    model_config = ConfigDict(extra="forbid")

    quote: str = Field(min_length=1)


class StructuredQaOutput(BaseModel):
    """Structured grounded QA output."""

    model_config = ConfigDict(extra="forbid")

    answerability: Answerability
    evidence: list[EvidenceQuote]
    answer: str = Field(min_length=1)
    confidence: ConfidenceLevel
