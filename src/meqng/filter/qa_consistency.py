from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import SentenceTransformer, util

from prompt import build_qa_question_answer_text
from .base import FilterDecision


@dataclass
class QAConsistencyFilter:
    """Keep candidates that stay semantically close to the original Q-A intent."""

    name: str = "qa_consistency"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    min_similarity: float = 0.70

    def __post_init__(self) -> None:
        self.model = SentenceTransformer(self.model_name)

    def check(self, knowledge: str, question: str, chosen: str, candidate: str) -> FilterDecision:
        ref = build_qa_question_answer_text(question=question, answer=chosen)
        cand = build_qa_question_answer_text(question=question, answer=candidate)
        embs = self.model.encode([ref, cand], convert_to_tensor=True, normalize_embeddings=True)
        sim = float(util.cos_sim(embs[0], embs[1]).item())
        keep = sim >= self.min_similarity
        return FilterDecision(
            keep=keep,
            reason="ok" if keep else "qa_semantic_shifted",
            meta={"similarity": sim, "min_similarity": self.min_similarity},
        )
