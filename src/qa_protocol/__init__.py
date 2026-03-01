"""Protocol helpers for structured grounded QA outputs."""

from .parsing import ParseResult, parse_structured_output, to_canonical_json, validate_structured_payload
from .prompting import build_infer_prompt, build_teacher_prompt
from .schema import Answerability, ConfidenceLevel, EvidenceQuote, StructuredQaOutput
from .spec import DEFAULT_PROTOCOL_SPEC, ProtocolSpec

__all__ = [
    "Answerability",
    "ConfidenceLevel",
    "EvidenceQuote",
    "StructuredQaOutput",
    "ProtocolSpec",
    "DEFAULT_PROTOCOL_SPEC",
    "ParseResult",
    "build_infer_prompt",
    "build_teacher_prompt",
    "parse_structured_output",
    "to_canonical_json",
    "validate_structured_payload",
]
