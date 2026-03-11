from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from project_config import PROJECT_SETTINGS


@dataclass(frozen=True)
class ProtocolSpec:
    """Configuration for one structured grounded-QA protocol version."""

    version: str = PROJECT_SETTINGS.protocol.version
    max_evidence_count: int = PROJECT_SETTINGS.protocol.max_evidence_count
    validation_tokenizer_name: str = PROJECT_SETTINGS.model.default_tokenizer_name
    max_completion_tokens: int = PROJECT_SETTINGS.protocol.max_completion_tokens
    prefilter_tokenizer_name: str = PROJECT_SETTINGS.model.default_tokenizer_name
    max_prompt_tokens: int = PROJECT_SETTINGS.protocol.max_prompt_tokens
    canonical_refusal: str = PROJECT_SETTINGS.protocol.canonical_refusal
    refusal_templates: Tuple[str, ...] = PROJECT_SETTINGS.protocol.refusal_templates
    field_order: Tuple[str, ...] = PROJECT_SETTINGS.protocol.field_order


DEFAULT_PROTOCOL_SPEC = ProtocolSpec()
