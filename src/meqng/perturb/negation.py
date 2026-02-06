from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import PerturbCandidate


_AUX_RE = re.compile(r"\b(is|are|was|were|has|have|had|can|could|will|would|should|may|might)\b", re.IGNORECASE)
_NOT_RE = re.compile(r"\bnot\b", re.IGNORECASE)


@dataclass
class NegationTogglePerturbator:
    """Toggle a simple negation around the first auxiliary verb occurrence."""
    name: str = "negation_toggle"

    def generate(
        self,
        knowledge: str,
        question: str,
        answer: str,
        entity_bank: Optional[Dict[str, Any]],
        max_candidates: int = 2,
        seed: int = 0,
    ) -> List[PerturbCandidate]:
        rng = random.Random(seed)
        if _NOT_RE.search(answer):
            # Remove the first 'not' occurrence.
            m = _NOT_RE.search(answer)
            new_text = answer[: m.start()] + answer[m.end():]
            new_text = re.sub(r"\s{2,}", " ", new_text).strip()
            if new_text != answer:
                return [PerturbCandidate(text=new_text, perturbator=self.name, meta={"op": "remove_not", "span": [m.start(), m.end()]})]
            return []

        m = _AUX_RE.search(answer)
        if not m:
            return []
        insert_pos = m.end()
        new_text = answer[:insert_pos] + " not" + answer[insert_pos:]
        if new_text != answer:
            return [PerturbCandidate(text=new_text, perturbator=self.name, meta={"op": "insert_not", "aux": m.group(0), "pos": insert_pos})]
        return []
