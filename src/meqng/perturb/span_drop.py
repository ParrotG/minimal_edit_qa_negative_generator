from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..ner import extract_numbers
from .base import PerturbCandidate


_SPAN_RE = re.compile(
    r"\b(?:in|on|at|by|from|during|after|before|with|for|under|over|between)\b"
    r"(?:\s+[A-Za-z0-9][A-Za-z0-9\-']*){1,6}",
    flags=re.IGNORECASE,
)


def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s.strip()


@dataclass
class SpanDropPerturbator:
    """
    Lightweight span-level perturbator.

    It removes one short prepositional phrase span, which is controllable and cheap for PoC.
    """

    name: str = "span_drop"
    min_words: int = 2
    max_words: int = 6

    def _find_spans(self, text: str) -> List[Tuple[str, int, int, int]]:
        spans: List[Tuple[str, int, int, int]] = []
        for m in _SPAN_RE.finditer(text):
            span_text = m.group(0)
            words = len(span_text.split())
            if words < self.min_words or words > self.max_words:
                continue
            spans.append((span_text, m.start(), m.end(), words))
        return spans

    def generate(
        self,
        knowledge: str,
        question: str,
        answer: str,
        entity_bank: Optional[Dict[str, Any]],
        max_candidates: int = 3,
        seed: int = 0,
    ) -> List[PerturbCandidate]:
        rng = random.Random(seed)
        spans = self._find_spans(answer)
        rng.shuffle(spans)

        out: List[PerturbCandidate] = []
        for span_text, start, end, words in spans:
            # Avoid direct numeric deletion if possible, because numeric perturbation is handled separately.
            if extract_numbers(span_text):
                continue
            new_text = _clean_text(answer[:start] + " " + answer[end:])
            if not new_text or new_text == _clean_text(answer):
                continue
            out.append(
                PerturbCandidate(
                    text=new_text,
                    perturbator=self.name,
                    meta={
                        "op": "drop_span",
                        "span_text": span_text,
                        "span": [start, end],
                        "words": words,
                    },
                )
            )
            if len(out) >= max_candidates:
                break
        return out
