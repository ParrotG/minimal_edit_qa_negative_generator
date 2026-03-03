from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import ValidationError

from .normalize import canonicalize_raw_text, canonicalize_structured_output, ordered_output_dict
from .schema import StructuredQaOutput
from .spec import DEFAULT_PROTOCOL_SPEC, ProtocolSpec


@dataclass(frozen=True)
class ParseResult:
    """Parse result for one raw model output."""

    ok: bool
    raw_text: str
    parsed: Optional[StructuredQaOutput] = None
    errors: tuple[str, ...] = field(default_factory=tuple)


def _extract_json_span(text: str) -> str:
    left = text.find("{")
    right = text.rfind("}")
    if left < 0 or right < 0 or right < left:
        raise ValueError("No JSON object found in model output.")
    return text[left : right + 1]


def parse_structured_output(
    text: str,
    spec: ProtocolSpec = DEFAULT_PROTOCOL_SPEC,
) -> ParseResult:
    """Parse one raw text into a validated structured output."""

    _ = spec
    raw_text = canonicalize_raw_text(text)
    if not raw_text:
        return ParseResult(ok=False, raw_text=raw_text, errors=("Empty model output.",))

    try:
        json_text = _extract_json_span(raw_text)
        parsed = StructuredQaOutput.model_validate_json(json_text)
        parsed = canonicalize_structured_output(parsed)
        return ParseResult(ok=True, raw_text=raw_text, parsed=parsed, errors=())
    except (ValueError, ValidationError, json.JSONDecodeError) as exc:
        return ParseResult(ok=False, raw_text=raw_text, errors=(str(exc),))


def validate_structured_payload(payload: Any) -> StructuredQaOutput:
    """Validate an already-decoded payload against the protocol schema."""

    return canonicalize_structured_output(StructuredQaOutput.model_validate(payload))


def to_canonical_json(
    output: StructuredQaOutput,
    spec: ProtocolSpec = DEFAULT_PROTOCOL_SPEC,
) -> str:
    """Serialize one structured output with stable formatting."""

    payload = ordered_output_dict(output=output, spec=spec)
    return json.dumps(payload, indent=2, ensure_ascii=False)
