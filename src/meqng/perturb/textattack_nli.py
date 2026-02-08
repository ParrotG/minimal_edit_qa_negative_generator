from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rapidfuzz.distance import Levenshtein
from sentence_transformers import SentenceTransformer, util

from ..ner import NERTagger, extract_numbers
from ..nli import NLIVerifier
from ..prompt import build_qa_premise
from .base import PerturbCandidate


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _norm_edit_distance(a: str, b: str) -> float:
    a2 = _norm_ws(a)
    b2 = _norm_ws(b)
    dist = Levenshtein.distance(a2, b2)
    return dist / max(1, max(len(a2), len(b2)))


@dataclass
class _ProtectedSpan:
    start: int
    end: int
    text: str


@dataclass
class TextAttackNLIFlipPerturbator:
    """
    Generate non-entailing candidates by TextAttack augmentation followed by NLI re-scoring.

    The class intentionally keeps TextAttack as an optional dependency:
    it is imported only when this perturbator is enabled.
    """

    verifier: NLIVerifier
    name: str = "textattack_nli_flip"
    augmenter: str = "embedding"
    pct_words_to_swap: float = 0.15
    transformations_per_example: int = 8
    search_calls: int = 6
    entail_threshold_neg: float = 0.35
    contradiction_ratio: float = 0.50
    min_norm_edit: float = 0.01
    max_norm_edit: float = 0.25
    semantic_model_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"
    min_semantic_similarity: float = 0.80
    protect_entities: bool = True
    protect_numbers: bool = True
    spacy_model: str = "en_core_web_trf"

    def __post_init__(self) -> None:
        self._augmenter = self._build_augmenter()
        self._semantic_model = SentenceTransformer(self.semantic_model_name) if self.semantic_model_name else None
        self._ner = NERTagger(self.spacy_model) if self.protect_entities else None

    def _build_augmenter(self) -> Any:
        try:
            from textattack.augmentation import EmbeddingAugmenter, WordNetAugmenter
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "TextAttack is not installed. Install optional dependency group first (e.g. pip install -e '.[attack]')."
            ) from exc

        aug = self.augmenter.strip().lower()
        if aug == "embedding":
            return EmbeddingAugmenter(
                pct_words_to_swap=self.pct_words_to_swap,
                transformations_per_example=self.transformations_per_example,
            )
        if aug == "wordnet":
            return WordNetAugmenter(
                pct_words_to_swap=self.pct_words_to_swap,
                transformations_per_example=self.transformations_per_example,
            )
        raise ValueError(f"Unknown TextAttack augmenter: {self.augmenter}. Available: embedding, wordnet")

    @staticmethod
    def _dedup_and_sort_spans(spans: Sequence[_ProtectedSpan]) -> List[_ProtectedSpan]:
        ordered = sorted(spans, key=lambda x: (x.start, x.end))
        out: List[_ProtectedSpan] = []
        for s in ordered:
            if not out:
                out.append(s)
                continue
            prev = out[-1]
            if s.start < prev.end:
                continue
            out.append(s)
        return out

    def _collect_protected_spans(self, answer: str) -> List[_ProtectedSpan]:
        spans: List[_ProtectedSpan] = []
        if self.protect_entities and self._ner is not None:
            for ent in self._ner.extract(answer):
                spans.append(_ProtectedSpan(start=ent.start, end=ent.end, text=ent.text))
        if self.protect_numbers:
            for raw, start, end in extract_numbers(answer):
                spans.append(_ProtectedSpan(start=start, end=end, text=raw))
        return self._dedup_and_sort_spans(spans)

    def _mask_protected(self, answer: str) -> Tuple[str, Dict[str, str]]:
        spans = self._collect_protected_spans(answer)
        if not spans:
            return answer, {}

        mapping: Dict[str, str] = {}
        masked = answer
        for idx, span in enumerate(reversed(spans)):
            key = f"__KEEP_{len(spans) - idx - 1}__"
            mapping[key] = span.text
            masked = masked[: span.start] + key + masked[span.end :]
        return masked, mapping

    @staticmethod
    def _unmask(candidate: str, mapping: Dict[str, str]) -> Optional[str]:
        out = candidate
        for key, val in mapping.items():
            if key not in out:
                return None
            out = out.replace(key, val)
        return out

    def _semantic_similarity(self, src: str, tgt: str) -> Optional[float]:
        if self._semantic_model is None:
            return None
        embs = self._semantic_model.encode([src, tgt], convert_to_tensor=True, normalize_embeddings=True)
        return float(util.cos_sim(embs[0], embs[1]).item())

    def _sample_by_target_mix(
        self,
        max_candidates: int,
        contradiction: List[Tuple[str, Dict[str, Any]]],
        neutral: List[Tuple[str, Dict[str, Any]]],
    ) -> List[PerturbCandidate]:
        contradiction.sort(key=lambda x: x[1]["nli_entail"], reverse=True)
        neutral.sort(key=lambda x: x[1]["nli_entail"], reverse=True)

        target_contra = int(round(max_candidates * self.contradiction_ratio))
        target_contra = max(0, min(max_candidates, target_contra))
        target_neutral = max_candidates - target_contra

        picked: List[Tuple[str, Dict[str, Any]]] = []
        picked.extend(contradiction[:target_contra])
        picked.extend(neutral[:target_neutral])

        if len(picked) < max_candidates:
            remain = contradiction[target_contra:] + neutral[target_neutral:]
            remain.sort(key=lambda x: x[1]["nli_entail"], reverse=True)
            picked.extend(remain[: max_candidates - len(picked)])

        return [
            PerturbCandidate(
                text=text,
                perturbator=self.name,
                meta=meta,
            )
            for text, meta in picked[:max_candidates]
        ]

    def generate(
        self,
        knowledge: str,
        question: str,
        answer: str,
        entity_bank: Optional[Dict[str, Any]],
        max_candidates: int = 3,
        seed: int = 0,
    ) -> List[PerturbCandidate]:
        if max_candidates <= 0:
            return []

        rng = random.Random(seed)
        premise = build_qa_premise(knowledge, question)
        masked_answer, mapping = self._mask_protected(answer)

        pool: Dict[str, Dict[str, Any]] = {}
        for call_idx in range(max(1, self.search_calls)):
            random.seed(rng.randint(0, 2**31 - 1) + call_idx)
            try:
                aug_texts = self._augmenter.augment(masked_answer)
            except Exception:
                continue

            for aug in aug_texts:
                restored = self._unmask(aug, mapping)
                if restored is None:
                    continue
                cand = _norm_ws(restored)
                if not cand or cand == _norm_ws(answer):
                    continue

                norm = _norm_edit_distance(answer, cand)
                if norm < self.min_norm_edit or norm > self.max_norm_edit:
                    continue

                sim = self._semantic_similarity(answer, cand)
                if sim is not None and sim < self.min_semantic_similarity:
                    continue

                pool[cand] = {"norm_edit": norm, "semantic_similarity": sim, "search_call": call_idx}

        if not pool:
            return []

        cands = list(pool.keys())
        scores = self.verifier.score([premise] * len(cands), cands)

        contradiction: List[Tuple[str, Dict[str, Any]]] = []
        neutral: List[Tuple[str, Dict[str, Any]]] = []
        for cand, sc in zip(cands, scores):
            if sc.entail > self.entail_threshold_neg:
                continue
            label = "contradiction" if sc.contradict >= sc.neutral else "neutral"
            meta = {
                **pool[cand],
                "nli_entail": sc.entail,
                "nli_neutral": sc.neutral,
                "nli_contradict": sc.contradict,
                "target_label": label,
                "entail_threshold_neg": self.entail_threshold_neg,
            }
            if label == "contradiction":
                contradiction.append((cand, meta))
            else:
                neutral.append((cand, meta))

        return self._sample_by_target_mix(
            max_candidates=max_candidates,
            contradiction=contradiction,
            neutral=neutral,
        )
