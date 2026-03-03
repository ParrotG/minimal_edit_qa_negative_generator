from __future__ import annotations

import html
import re
import unicodedata
from typing import Any, Dict

from .schema import StructuredQaOutput
from .spec import DEFAULT_PROTOCOL_SPEC, ProtocolSpec


_WHITESPACE_RE = re.compile(r"\s+")
_ZERO_WIDTH_RE = re.compile(r"[\ufeff\u200b\u200c\u200d\u2060]")


def canonicalize_raw_text(text: str) -> str:
    """Canonicalize raw model output without collapsing internal whitespace."""

    value = html.unescape(str(text or ""))
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = _ZERO_WIDTH_RE.sub("", value)
    return value.strip()


def canonicalize_field_text(text: str) -> str:
    """Canonicalize one structured string field while preserving internal spacing."""

    return canonicalize_raw_text(text)


def normalize_text(text: str) -> str:
    """Normalize whitespace for stable comparisons."""

    return _WHITESPACE_RE.sub(" ", canonicalize_field_text(text))


def normalize_quote(text: str) -> str:
    """Normalize quote text for substring matching."""

    return normalize_text(text)


def canonicalize_structured_output(output: StructuredQaOutput) -> StructuredQaOutput:
    """Canonicalize all free-text fields in one structured output."""

    return output.model_copy(
        update={
            "evidence": [
                item.model_copy(update={"quote": canonicalize_field_text(item.quote)})
                for item in output.evidence
            ],
            "rationale": canonicalize_field_text(output.rationale),
            "answer": canonicalize_field_text(output.answer),
        }
    )


def ordered_output_dict(
    output: StructuredQaOutput,
    spec: ProtocolSpec = DEFAULT_PROTOCOL_SPEC,
) -> Dict[str, Any]:
    """Convert a structured output into a stable ordered dictionary."""

    payload = canonicalize_structured_output(output).model_dump(mode="json")
    return {field: payload[field] for field in spec.field_order}
