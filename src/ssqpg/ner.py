from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

import spacy


@dataclass(frozen=True)
class EntitySpan:
    """Entity span extracted from text."""

    text: str
    label: str
    start: int
    end: int


class NERTagger:
    """Lightweight wrapper around spaCy named-entity recognition."""

    def __init__(self, model_name: str = "en_core_web_trf") -> None:
        self.nlp = spacy.load(model_name)

    def extract(self, text: str) -> List[EntitySpan]:
        """Extract entity spans from text."""

        doc = self.nlp(text)
        spans: List[EntitySpan] = []
        for ent in doc.ents:
            spans.append(EntitySpan(text=ent.text, label=ent.label_, start=ent.start_char, end=ent.end_char))
        return spans


_NUMBER_RE = re.compile(r"(?<!\w)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:%|)(?!\w)")


def extract_numbers(text: str) -> List[Tuple[str, int, int]]:
    """Extract numeric spans with offsets."""

    return [(m.group(0), m.start(), m.end()) for m in _NUMBER_RE.finditer(text)]
