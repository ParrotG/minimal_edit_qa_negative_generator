from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..ner import NERTagger
from .base import PerturbCandidate


def _match_case_like(src: str, tgt: str) -> str:
    """Match the casing style of `src` when applying `tgt`."""
    if src.isupper():
        return tgt.upper()
    if src.islower():
        return tgt.lower()
    if src.istitle():
        return tgt.title()
    return tgt


def _choose_replacement(src: str, items: List[Dict[str, Any]], rng: random.Random) -> Optional[str]:
    """Choose a replacement entity with mild length constraints to reduce stylistic drift."""
    src_len = len(src)
    # Prefer candidates with similar surface length.
    candidates = [it["text"] for it in items if abs(len(it["text"]) - src_len) <= 4 and it["text"] != src]
    if not candidates:
        candidates = [it["text"] for it in items if it["text"] != src]
    if not candidates:
        return None
    return rng.choice(candidates)


@dataclass
class EntitySwapPerturbator:
    """Replace one named entity in the answer with another entity of the same type."""
    name: str = "entity_swap"
    spacy_model: str = "en_core_web_trf"

    def __post_init__(self) -> None:
        self.ner = NERTagger(self.spacy_model)

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
        if not entity_bank:
            return []

        ents = self.ner.extract(answer)
        # Filter to "contentful" entity types; you can expand this list later.
        ents = [e for e in ents if e.label in entity_bank and e.label not in {"DATE", "TIME", "CARDINAL"}]
        rng.shuffle(ents)

        out: List[PerturbCandidate] = []
        for ent in ents:
            items = entity_bank.get(ent.label, [])
            repl = _choose_replacement(ent.text, items, rng)
            if not repl:
                continue
            repl = _match_case_like(ent.text, repl)
            new_text = answer[: ent.start] + repl + answer[ent.end :]
            if new_text != answer:
                out.append(
                    PerturbCandidate(
                        text=new_text,
                        perturbator=self.name,
                        meta={"label": ent.label, "orig": ent.text, "repl": repl, "span": [ent.start, ent.end]},
                    )
                )
            if len(out) >= max_candidates:
                break
        return out
