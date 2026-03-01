from __future__ import annotations

from .normalize import normalize_text
from .spec import DEFAULT_PROTOCOL_SPEC, ProtocolSpec


def is_refusal_template(text: str, spec: ProtocolSpec = DEFAULT_PROTOCOL_SPEC) -> bool:
    """Check whether a refusal answer matches one of the allowed templates."""

    normalized = normalize_text(text)
    return normalized in {normalize_text(item) for item in spec.refusal_templates}


def canonical_refusal(spec: ProtocolSpec = DEFAULT_PROTOCOL_SPEC) -> str:
    """Return the canonical refusal text used for training targets."""

    return spec.canonical_refusal
