from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .base import FilterDecision
from ..nli import NLIVerifier
from ..prompt import build_qa_premise


@dataclass
class NLIFlipFilter:
    """Require the chosen answer to be entailed and the candidate to be non-entailed."""
    name: str = "nli_flip"
    verifier: NLIVerifier = None  # injected
    entail_threshold_pos: float = 0.60
    entail_threshold_neg: float = 0.30

    def check(self, knowledge: str, question: str, chosen: str, candidate: str) -> FilterDecision:
        premise = build_qa_premise(knowledge, question)
        pos = self.verifier.score([premise], [chosen])[0]
        neg = self.verifier.score([premise], [candidate])[0]

        pos_ok = pos.entail >= self.entail_threshold_pos
        neg_ok = neg.entail <= self.entail_threshold_neg

        keep = bool(pos_ok and neg_ok)
        reason = "ok" if keep else "nli_not_flipped"
        meta: Dict[str, Any] = {
            "pos_entail": pos.entail,
            "pos_neutral": pos.neutral,
            "pos_contradict": pos.contradict,
            "cand_entail": neg.entail,
            "cand_neutral": neg.neutral,
            "cand_contradict": neg.contradict,
        }
        return FilterDecision(keep, reason, meta)
