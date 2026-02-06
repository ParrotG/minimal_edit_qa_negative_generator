from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import spacy


@dataclass(frozen=True)
class EntitySpan:
    """A detected entity span."""
    text: str
    label: str
    start: int
    end: int


class NERTagger:
    """spaCy-based NER wrapper."""

    def __init__(self, model: str = "en_core_web_trf") -> None:
        self.nlp = spacy.load(model)

    def extract(self, text: str) -> List[EntitySpan]:
        """Extract entities from a text."""
        doc = self.nlp(text)
        spans: List[EntitySpan] = []
        for ent in doc.ents:
            spans.append(EntitySpan(ent.text, ent.label_, ent.start_char, ent.end_char))
        return spans


_NUMBER_RE = re.compile(r"(?<!\w)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:%|)(?!\w)")


def extract_numbers(text: str) -> List[Tuple[str, int, int]]:
    """Extract numeric substrings with span offsets."""
    return [(m.group(0), m.start(), m.end()) for m in _NUMBER_RE.finditer(text)]


def build_entity_bank(texts: Iterable[str], ner: NERTagger, min_count: int = 2) -> Dict[str, List[Dict[str, int | str]]]:
    """Build a label->entities bank from corpus texts."""
    counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for t in texts:
        for ent in ner.extract(t):
            norm = ent.text.strip()
            if not norm:
                continue
            counts[ent.label][norm] += 1

    bank: Dict[str, List[Dict[str, int | str]]] = {}
    for label, ctr in counts.items():
        items = [{"text": txt, "count": int(c)} for txt, c in ctr.most_common() if c >= min_count]
        if items:
            bank[label] = items
    return bank
