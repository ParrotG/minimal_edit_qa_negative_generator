from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TokenUsage:
    """Token usage metadata for one generation result."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    source: str


@dataclass(frozen=True)
class LocalGenerationResult:
    """One local-generation result with optional token usage."""

    text: str
    token_usage: Optional[TokenUsage] = None
