from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass(frozen=True)
class FilterDecision:
    """Filter decision with optional reason."""
    keep: bool
    reason: str
    meta: Dict[str, Any]


class CandidateFilter(Protocol):
    """Filter protocol."""
    name: str

    def check(self, knowledge: str, question: str, chosen: str, candidate: str) -> FilterDecision:
        """Return whether the candidate should be kept."""
        ...
