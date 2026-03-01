from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ProtocolCheckReport:
    """Protocol-check result for one structured output."""

    ok: bool
    issues: list[str] = field(default_factory=list)
    answer_length: int = 0
    evidence_count: int = 0


@dataclass
class EvidenceCheckReport:
    """Evidence-check result for one structured output."""

    ok: bool
    issues: list[str] = field(default_factory=list)
    quote_match_rate: float = 0.0
    unique_quotes: int = 0


@dataclass
class CorrectnessCheckReport:
    """Correctness result relative to the reference answer."""

    ok: bool
    exact_match: bool = False
    token_f1: float = 0.0
    semantic_match: bool = False
    issues: list[str] = field(default_factory=list)


@dataclass
class SemanticCheckReport:
    """Semantic-support result derived from verifier signals."""

    ok: Optional[bool]
    supported: Optional[bool]
    answer_type_ok: Optional[bool]
    refusal_ok: Optional[bool]
    details: Dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Combined validation result for one candidate output."""

    parse_ok: bool
    overall_ok: bool
    protocol: ProtocolCheckReport
    evidence: EvidenceCheckReport
    correctness: Optional[CorrectnessCheckReport] = None
    semantics: Optional[SemanticCheckReport] = None
    answerability_match: Optional[bool] = None
    derived_confidence: Optional[str] = None
    selection_score: float = 0.0
    issues: list[str] = field(default_factory=list)
