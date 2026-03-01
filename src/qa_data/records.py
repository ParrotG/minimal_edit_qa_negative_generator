from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ContextDocument:
    """One context document from the source dataset."""

    title: str
    sentences: tuple[str, ...]


@dataclass
class SupportingSentence:
    """One supporting sentence with its support window."""

    title: str
    sent_id: int
    sentence: str
    window_sentences: tuple[str, ...]
    window_text: str


@dataclass
class QaExample:
    """Canonical grounded-QA example record."""

    id: str
    source_id: str
    variant_id: str
    split: Optional[str]
    question: str
    knowledge: str
    reference_answer: str
    answerability_label: str
    difficulty: str
    supporting_sentences: tuple[SupportingSentence, ...] = ()
    context_documents: tuple[ContextDocument, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)
