from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass(frozen=True)
class PerturbCandidate:
    """A single perturbed answer candidate."""
    text: str
    perturbator: str
    meta: Dict[str, Any]


class Perturbator(Protocol):
    """Perturbator protocol."""

    name: str

    def generate(
        self,
        knowledge: str,
        question: str,
        answer: str,
        entity_bank: Optional[Dict[str, Any]],
        max_candidates: int,
        seed: int,
    ) -> List[PerturbCandidate]:
        """Generate perturbation candidates from an answer."""
        ...
