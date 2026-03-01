from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from qa_protocol.refusal import is_refusal_template
from qa_protocol.schema import Answerability, StructuredQaOutput

from .config import JudgeConfig, NLIConfig
from .judge import AnswerJudge
from .nli import NLIVerifier


class StructuredAnswerJudge:
    """Adapter that reuses the plain QA judge for structured grounded-QA outputs."""

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
    def _selected_knowledge(row: Dict[str, Any], output: StructuredQaOutput) -> str:
        quotes = [str(item.quote).strip() for item in output.evidence if str(item.quote).strip()]
        if quotes:
            return "\n".join(quotes)
        fallback = str(row.get("support_window_knowledge") or row.get("knowledge") or "").strip()
        return fallback

    def judge_rows(self, rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Judge a batch of structured outputs."""

        out: List[Dict[str, Any]] = []
        answerable_indices: List[int] = []
        plain_rows: List[Dict[str, str]] = []
        selected_knowledge_by_index: Dict[int, str] = {}

        for idx, row in enumerate(rows):
            output = self._coerce_output(row.get("parsed_output"))
            selected_knowledge = self._selected_knowledge(row, output)
            selected_knowledge_by_index[idx] = selected_knowledge

            if output.answerability == Answerability.UNANSWERABLE:
                refusal_ok = is_refusal_template(output.answer)
                out.append(
                    {
                        **row,
                        "structured_judge": {
                            "ok": refusal_ok,
                            "refusal_ok": refusal_ok,
                            "issues": [] if refusal_ok else ["Refusal answer is not in the allowed template set."],
                            "selected_knowledge": selected_knowledge,
                            "answer_judge": {
                                "is_correct": None,
                                "is_abstain": None,
                                "qa_consistent": None,
                            },
                        },
                    }
                )
                continue

            answerable_indices.append(idx)
            plain_rows.append(
                {
                    "knowledge": selected_knowledge,
                    "question": str(row.get("question") or "").strip(),
                    "answer": output.answer,
                }
            )
            out.append(dict(row))

        judged_answerable, metrics = self.answer_judge.judge(plain_rows) if plain_rows else ([], {"num_rows": 0.0})
        for judged_idx, source_idx in enumerate(answerable_indices):
            answer_judge = judged_answerable[judged_idx].get("judge") or {}
            out[source_idx]["structured_judge"] = {
                "ok": bool(answer_judge.get("is_correct")),
                "refusal_ok": None,
                "issues": [] if bool(answer_judge.get("is_correct")) else ["Answer judge did not mark the answer as supported."],
                "selected_knowledge": selected_knowledge_by_index[source_idx],
                "answer_judge": answer_judge,
            }

        return out, metrics
