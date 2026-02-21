from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from sentence_transformers import SentenceTransformer, util

from .config import JudgeConfig
from .nli import NLIScores, NLIVerifier
from .prompt import build_qa_premise


class AnswerJudge:
    """Judge sampled answers with NLI evidence support and optional semantic QA consistency."""

    def __init__(
        self,
        cfg: JudgeConfig,
        primary_verifier: NLIVerifier,
        secondary_verifier: Optional[NLIVerifier] = None,
    ) -> None:
        self.cfg = cfg
        self.primary = primary_verifier
        self.secondary = secondary_verifier

        vote_mode = cfg.vote_mode.strip().lower()
        if vote_mode not in {"primary", "and", "or"}:
            raise ValueError("vote_mode must be one of: primary, and, or")
        self.vote_mode = vote_mode

        self.qa_model: Optional[SentenceTransformer] = None
        if cfg.qa_similarity_model_name:
            self.qa_model = SentenceTransformer(cfg.qa_similarity_model_name)

    def _qa_similarity_scores(self, rows: Sequence[Dict[str, Any]]) -> List[Optional[float]]:
        if self.qa_model is None:
            return [None] * len(rows)

        valid_indices: List[int] = []
        refs: List[str] = []
        cands: List[str] = []

        for i, row in enumerate(rows):
            reference = str(row.get("reference_answer") or "").strip()
            if not reference:
                continue
            valid_indices.append(i)
            refs.append(f"Question: {row['question']}\nAnswer: {reference}")
            cands.append(f"Question: {row['question']}\nAnswer: {row['answer']}")

        out: List[Optional[float]] = [None] * len(rows)
        if not valid_indices:
            return out

        embs = self.qa_model.encode(refs + cands, convert_to_tensor=True, normalize_embeddings=True)
        n = len(valid_indices)
        for j, idx in enumerate(valid_indices):
            out[idx] = float(util.cos_sim(embs[j], embs[n + j]).item())
        return out

    def _vote(self, primary_supported: bool, secondary_supported: Optional[bool]) -> bool:
        if self.vote_mode == "primary":
            return primary_supported
        if secondary_supported is None:
            return primary_supported
        if self.vote_mode == "and":
            return bool(primary_supported and secondary_supported)
        return bool(primary_supported or secondary_supported)

    @staticmethod
    def _index_scores(indices: List[int], scores: List[NLIScores], total_len: int) -> List[Optional[NLIScores]]:
        out: List[Optional[NLIScores]] = [None] * total_len
        for idx, sc in zip(indices, scores):
            out[idx] = sc
        return out

    def judge(self, rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Run NLI-based faithfulness judgment on sampled answers."""

        if not rows:
            return [], {
                "num_rows": 0.0,
                "num_correct": 0.0,
                "correct_rate": 0.0,
                "num_with_reference": 0.0,
            }

        premises = [build_qa_premise(r["knowledge"], r["question"]) for r in rows]
        answers = [r["answer"] for r in rows]

        p_cand = self.primary.score(premises, answers)

        ref_indices: List[int] = []
        ref_premises: List[str] = []
        references: List[str] = []
        for i, row in enumerate(rows):
            reference = str(row.get("reference_answer") or "").strip()
            if not reference:
                continue
            ref_indices.append(i)
            ref_premises.append(premises[i])
            references.append(reference)

        p_ref_indexed: List[Optional[NLIScores]] = [None] * len(rows)
        if ref_indices:
            p_ref_scores = self.primary.score(ref_premises, references)
            p_ref_indexed = self._index_scores(ref_indices, p_ref_scores, total_len=len(rows))

        s_cand = None
        s_ref_indexed: List[Optional[NLIScores]] = [None] * len(rows)
        if self.secondary is not None:
            s_cand = self.secondary.score(premises, answers)
            if ref_indices:
                s_ref_scores = self.secondary.score(ref_premises, references)
                s_ref_indexed = self._index_scores(ref_indices, s_ref_scores, total_len=len(rows))

        qa_sims = self._qa_similarity_scores(rows)

        judged: List[Dict[str, Any]] = []
        correct = 0
        num_with_reference = 0

        for i, row in enumerate(rows):
            p_c = p_cand[i]
            p_r = p_ref_indexed[i]

            p_c_supported = p_c.entail >= self.cfg.candidate_entail_threshold and p_c.contradict <= self.cfg.candidate_contradict_threshold
            p_r_supported = True if p_r is None else p_r.entail >= self.cfg.reference_entail_threshold
            has_reference = p_r is not None
            if has_reference:
                num_with_reference += 1

            s_c_supported: Optional[bool] = None
            s_r_supported: Optional[bool] = None
            if s_cand is not None:
                s_c = s_cand[i]
                s_c_supported = s_c.entail >= self.cfg.candidate_entail_threshold and s_c.contradict <= self.cfg.candidate_contradict_threshold

                s_r = s_ref_indexed[i]
                if s_r is not None:
                    s_r_supported = s_r.entail >= self.cfg.reference_entail_threshold

            voted_supported = self._vote(p_c_supported, s_c_supported)
            qa_sim = qa_sims[i]
            qa_consistent = qa_sim is None or qa_sim >= self.cfg.qa_similarity_min

            # Strict gate: candidate support + optional reference support + optional QA semantic consistency.
            is_correct = bool(voted_supported and p_r_supported and qa_consistent)
            if is_correct:
                correct += 1

            judge_payload: Dict[str, Any] = {
                "vote_mode": self.vote_mode,
                "has_reference": has_reference,
                "candidate_entail_primary": p_c.entail,
                "candidate_neutral_primary": p_c.neutral,
                "candidate_contradict_primary": p_c.contradict,
                "reference_entail_primary": None if p_r is None else p_r.entail,
                "reference_neutral_primary": None if p_r is None else p_r.neutral,
                "reference_contradict_primary": None if p_r is None else p_r.contradict,
                "candidate_supported_primary": p_c_supported,
                "reference_supported_primary": p_r_supported,
                "candidate_supported_secondary": s_c_supported,
                "reference_supported_secondary": s_r_supported,
                "candidate_supported": voted_supported,
                "qa_similarity": qa_sim,
                "qa_similarity_min": self.cfg.qa_similarity_min if qa_sim is not None else None,
                "qa_consistent": qa_consistent,
                "is_correct": is_correct,
            }
            judged.append({**row, "judge": judge_payload})

        metrics = {
            "num_rows": float(len(rows)),
            "num_correct": float(correct),
            "correct_rate": float(correct / max(1, len(rows))),
            "num_with_reference": float(num_with_reference),
        }
        return judged, metrics
