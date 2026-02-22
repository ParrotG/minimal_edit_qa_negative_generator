from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import JudgeConfig
from .nli import NLIScores, NLIVerifier
from .qa_consistency import NERTagger, check_answer_type
from prompt import build_qa_premise


class AnswerJudge:
    """Judge sampled answers with calibrated NLI margins and optional QA type checks."""

    def __init__(self, cfg: JudgeConfig, verifier: NLIVerifier) -> None:
        self.cfg = cfg
        self.verifier = verifier

        if self.cfg.temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if self.cfg.reject_band_half_width < 0.0:
            raise ValueError("reject_band_half_width must be >= 0")

        self.ner: Optional[NERTagger] = None
        if self.cfg.qa_check_answer_type:
            self.ner = NERTagger(self.cfg.qa_spacy_model)

    @staticmethod
    def _softmax(xs: Sequence[float]) -> List[float]:
        m = max(xs)
        exps = [math.exp(x - m) for x in xs]
        s = sum(exps)
        if s <= 0.0:
            return [1.0 / len(xs)] * len(xs)
        return [x / s for x in exps]

    def _temperature_calibrated(self, sc: NLIScores) -> Dict[str, float]:
        eps = 1e-12
        logp_e = math.log(max(sc.entail, eps))
        logp_n = math.log(max(sc.neutral, eps))
        logp_c = math.log(max(sc.contradict, eps))

        scaled = [logp_e / self.cfg.temperature, logp_n / self.cfg.temperature, logp_c / self.cfg.temperature]
        p_e, p_n, p_c = self._softmax(scaled)
        margin = scaled[0] - max(scaled[1], scaled[2])

        return {
            "entail": float(p_e),
            "neutral": float(p_n),
            "contradict": float(p_c),
            "margin": float(margin),
            "argmax_prob": float(max(p_e, p_n, p_c)),
            "argmax_label": "entail" if p_e >= p_n and p_e >= p_c else ("neutral" if p_n >= p_c else "contradict"),
        }

    def _nli_decision_full(self, margin: float) -> bool:
        # Method B: margin >= tau => supported.
        return bool(margin >= self.cfg.full_margin_threshold)

    def _nli_decision_reject(self, margin: float) -> Optional[bool]:
        # Method D: abstain in boundary band; otherwise thresholded decision.
        if abs(margin - self.cfg.reject_margin_threshold) <= self.cfg.reject_band_half_width:
            return None
        return bool(margin >= self.cfg.reject_margin_threshold)

    def _finalize_with_qa_gate(self, *, nli_full: bool, nli_reject: Optional[bool], qa_consistent: bool) -> Tuple[bool, Optional[bool], bool]:
        qa_forced_negative = bool(self.cfg.qa_fail_as_negative and not qa_consistent)
        if qa_forced_negative:
            return False, False, True
        return nli_full, nli_reject, False

    def judge(self, rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Run calibrated NLI judgment and return per-row payloads plus aggregate metrics."""

        if not rows:
            return [], {
                "num_rows": 0.0,
                "num_full_yes": 0.0,
                "num_full_no": 0.0,
                "num_reject_yes": 0.0,
                "num_reject_no": 0.0,
                "num_reject_abstain": 0.0,
                "abstain_rate": 0.0,
            }

        premises = [build_qa_premise(str(r["knowledge"]), str(r["question"])) for r in rows]
        answers = [str(r["answer"]) for r in rows]
        raw_scores = self.verifier.score(premises, answers)

        judged: List[Dict[str, Any]] = []
        num_full_yes = 0
        num_reject_yes = 0
        num_reject_no = 0
        num_reject_abstain = 0

        for row, raw in zip(rows, raw_scores):
            calib = self._temperature_calibrated(raw)
            margin = float(calib["margin"])

            nli_full = self._nli_decision_full(margin)
            nli_reject = self._nli_decision_reject(margin)

            qa_info = check_answer_type(question=str(row["question"]), answer=str(row["answer"]), ner=self.ner)
            qa_consistent = bool(qa_info.get("answer_type_ok", True))

            final_full, final_reject, qa_forced_negative = self._finalize_with_qa_gate(
                nli_full=nli_full,
                nli_reject=nli_reject,
                qa_consistent=qa_consistent,
            )

            if final_full:
                num_full_yes += 1

            if final_reject is None:
                num_reject_abstain += 1
                decision_reject = "abstain"
            elif bool(final_reject):
                num_reject_yes += 1
                decision_reject = "yes"
            else:
                num_reject_no += 1
                decision_reject = "no"

            judge_payload: Dict[str, Any] = {
                "temperature": float(self.cfg.temperature),
                "margin": margin,
                "nli_raw": {
                    "entail": float(raw.entail),
                    "neutral": float(raw.neutral),
                    "contradict": float(raw.contradict),
                },
                "nli_calibrated": calib,
                "full_binary": {
                    "method": "logit_margin_threshold",
                    "margin_threshold": float(self.cfg.full_margin_threshold),
                    "supported_by_nli": bool(nli_full),
                    "decision": "yes" if final_full else "no",
                },
                "reject_aware": {
                    "method": "margin_with_band_reject",
                    "margin_threshold": float(self.cfg.reject_margin_threshold),
                    "band_half_width": float(self.cfg.reject_band_half_width),
                    "nli_decision": (
                        "abstain" if nli_reject is None else ("yes" if bool(nli_reject) else "no")
                    ),
                    "decision": decision_reject,
                },
                "qa_consistency": qa_info,
                "qa_consistent": qa_consistent,
                "qa_forced_negative": qa_forced_negative,
                "is_correct": bool(final_reject is True),
                "is_abstain": bool(final_reject is None),
            }
            judged.append({**row, "judge": judge_payload})

        metrics = {
            "num_rows": float(len(rows)),
            "num_full_yes": float(num_full_yes),
            "num_full_no": float(len(rows) - num_full_yes),
            "num_reject_yes": float(num_reject_yes),
            "num_reject_no": float(num_reject_no),
            "num_reject_abstain": float(num_reject_abstain),
            "abstain_rate": float(num_reject_abstain / max(1, len(rows))),
        }
        return judged, metrics
