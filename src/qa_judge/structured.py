from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from qa_protocol.schema import Answerability, StructuredQaOutput

from .config import JudgeConfig, NLIConfig
from .judge import AnswerJudge
from .nli import NLIVerifier


class StructuredAnswerJudge:
    """Adapter that evaluates evidence-grounded structured outputs with the QA judge."""

    def __init__(self, answer_judge: AnswerJudge) -> None:
        self.answer_judge = answer_judge

    @classmethod
    def from_defaults(
        cls,
        *,
        nli_config: Optional[NLIConfig] = None,
        judge_config: Optional[JudgeConfig] = None,
    ) -> "StructuredAnswerJudge":
        """Build a structured-output judge from the default QA judge components."""

        resolved_nli = nli_config or NLIConfig()
        resolved_judge = judge_config or JudgeConfig()
        verifier = NLIVerifier(
            model_name=resolved_nli.model_name,
            device=resolved_nli.device,
            batch_size=resolved_nli.batch_size,
            max_length=resolved_nli.max_length,
            fp16=resolved_nli.fp16,
        )
        return cls(answer_judge=AnswerJudge(cfg=resolved_judge, verifier=verifier))

    @staticmethod
    def _coerce_output(payload: Any) -> StructuredQaOutput:
        if isinstance(payload, StructuredQaOutput):
            return payload
        return StructuredQaOutput.model_validate(payload)

    @staticmethod
    def _selected_evidence_text(output: StructuredQaOutput) -> str | None:
        quotes = [str(item.quote).strip() for item in output.evidence if str(item.quote).strip()]
        if not quotes:
            return None
        return "\n".join(quotes)

    @staticmethod
    def _build_semantic_hypothesis(output: StructuredQaOutput) -> str:
        rationale = str(output.rationale or "").strip()
        answer = str(output.answer or "").strip()
        return f"{rationale}\nTherefore the answer is {answer}"

    @staticmethod
    def _semantic_decision(answer_judge: Dict[str, Any], decision_source: str) -> Optional[str]:
        if decision_source == "reject_aware":
            return str((((answer_judge or {}).get("reject_aware") or {}).get("decision")) or "abstain")
        return str((((answer_judge or {}).get("full_binary") or {}).get("decision")) or "no")

    def judge_rows(
        self,
        rows: Sequence[Dict[str, Any]],
        *,
        decision_source: str = "full_binary",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Judge a batch of structured outputs."""

        if decision_source not in {"full_binary", "reject_aware"}:
            raise ValueError(f"Unsupported semantic decision source: {decision_source}")

        out: List[Dict[str, Any]] = []
        answerable_indices: List[int] = []
        plain_rows: List[Dict[str, str]] = []
        selected_knowledge_by_index: Dict[int, Optional[str]] = {}
        skipped_reports: Dict[int, Dict[str, Any]] = {}

        for idx, row in enumerate(rows):
            output = self._coerce_output(row.get("parsed_output"))
            selected_knowledge = self._selected_evidence_text(output)
            selected_knowledge_by_index[idx] = selected_knowledge

            if output.answerability == Answerability.UNANSWERABLE:
                out.append(
                    {
                        **row,
                        "structured_judge": {
                            "ok": None,
                            "supported": None,
                            "refusal_ok": None,
                            "selected_knowledge": None,
                            "hypothesis": None,
                            "decision_source": decision_source,
                            "decision": None,
                            "margin": None,
                            "issues": [],
                            "answer_judge": {},
                        },
                    }
                )
                continue

            if selected_knowledge is None:
                skipped_reports[idx] = {
                    "ok": None,
                    "supported": None,
                    "refusal_ok": None,
                    "selected_knowledge": None,
                    "hypothesis": None,
                    "decision_source": decision_source,
                    "decision": None,
                    "margin": None,
                    "issues": ["Semantic check skipped because no valid evidence quotes are available."],
                    "answer_judge": {},
                }
                out.append(dict(row))
                continue

            answerable_indices.append(idx)
            plain_rows.append(
                {
                    "knowledge": selected_knowledge,
                    "question": str(row.get("question") or "").strip(),
                    "answer": self._build_semantic_hypothesis(output),
                }
            )
            out.append(dict(row))

        judged_answerable, metrics = self.answer_judge.judge(plain_rows) if plain_rows else ([], {"num_rows": 0.0})
        for judged_idx, source_idx in enumerate(answerable_indices):
            answer_judge = judged_answerable[judged_idx].get("judge") or {}
            decision = self._semantic_decision(answer_judge, decision_source)
            out[source_idx]["structured_judge"] = {
                "ok": True if decision == "yes" else (False if decision == "no" else None),
                "supported": bool((answer_judge.get("full_binary") or {}).get("supported_by_nli")),
                "refusal_ok": None,
                "selected_knowledge": selected_knowledge_by_index[source_idx],
                "hypothesis": plain_rows[judged_idx]["answer"],
                "decision_source": decision_source,
                "decision": decision,
                "margin": answer_judge.get("margin"),
                "issues": [] if decision == "yes" else ["Semantic judge did not mark the rationale-answer chain as supported."],
                "answer_judge": answer_judge,
            }

        for source_idx, payload in skipped_reports.items():
            out[source_idx]["structured_judge"] = payload

        return out, metrics
