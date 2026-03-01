from __future__ import annotations

import re
from typing import Any, Dict

from .schema import StructuredQaOutput
from .spec import DEFAULT_PROTOCOL_SPEC, ProtocolSpec


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize whitespace for stable comparisons."""

    return _WHITESPACE_RE.sub(" ", str(text or "").strip())


def normalize_quote(text: str) -> str:
    """Normalize quote text for substring matching."""

    return normalize_text(text)


def ordered_output_dict(
    output: StructuredQaOutput,
    spec: ProtocolSpec = DEFAULT_PROTOCOL_SPEC,
) -> Dict[str, Any]:
    """Convert a structured output into a stable ordered dictionary."""

    payload = output.model_dump(mode="json")
    return {field: payload[field] for field in spec.field_order}
