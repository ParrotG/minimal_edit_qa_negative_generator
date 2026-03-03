"""Validation and selection helpers for structured grounded QA outputs."""

from .correctness import CorrectnessConfig, check_answer_correctness
from .evidence import check_evidence_against_supporting_facts, check_evidence_quotes
from .protocol import check_protocol_constraints
from .report import (
    CorrectnessCheckReport,
    EvidenceCheckReport,
    ProtocolCheckReport,
    SemanticCheckReport,
    ValidationReport,
)
from .selection import assign_quantile_confidence_labels, score_validation_report, select_best_candidates
from .semantics import evaluate_structured_semantics
from .unanswerable_prefilter import UnanswerablePrefilterReport, build_reference_answer_hypothesis, check_reference_answer_unsupported
from .unanswerable_selection import select_first_valid_unanswerable_candidate

__all__ = [
    "CorrectnessConfig",
    "CorrectnessCheckReport",
    "EvidenceCheckReport",
    "ProtocolCheckReport",
    "SemanticCheckReport",
    "ValidationReport",
    "check_answer_correctness",
    "check_evidence_against_supporting_facts",
    "check_evidence_quotes",
    "check_protocol_constraints",
    "assign_quantile_confidence_labels",
    "score_validation_report",
    "select_best_candidates",
    "evaluate_structured_semantics",
    "UnanswerablePrefilterReport",
    "build_reference_answer_hypothesis",
    "check_reference_answer_unsupported",
    "select_first_valid_unanswerable_candidate",
]
