"""Protocol helpers for structured grounded QA outputs."""

from .normalize import canonicalize_field_text, canonicalize_raw_text, canonicalize_structured_output
from .parsing import ParseResult, parse_structured_output, to_canonical_json, validate_structured_payload
from .prompting import build_infer_prompt, build_teacher_prompt
from .schema import Answerability, ConfidenceLevel, EvidenceQuote, StructuredQaOutput
from .spec import DEFAULT_PROTOCOL_SPEC, ProtocolSpec
from .token_budget import count_text_tokens, count_text_tokens_batch

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
    "canonicalize_field_text",
    "canonicalize_raw_text",
    "canonicalize_structured_output",
    "count_text_tokens",
    "count_text_tokens_batch",
    "parse_structured_output",
    "to_canonical_json",
    "validate_structured_payload",
]
