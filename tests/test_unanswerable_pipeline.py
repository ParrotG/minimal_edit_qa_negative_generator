from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from grounded_qa.config import TeacherGenerationConfig, ValidationConfig
from grounded_qa.workflow import generate_teacher_candidates, validate_mixed_teacher_candidates
from qa_checks.source_prefilter import prefilter_mixed_examples
from qa_checks.unanswerable_prefilter import (
    build_reference_answer_hypothesis,
    check_reference_answer_unsupported,
    flip_yes_no_answer,
)
from qa_data.construct import ConstructionConfig
from qa_data.records import ContextDocument, QaExample, SupportingSentence
from qa_data.tagging import AnswerabilitySplitConfig, DataSplitConfig, assign_answerability_split_name, assign_data_split_name
from qa_data.unanswerable import UnanswerableBuildConfig, build_examples_from_tagged_row, build_unanswerable_from_answerable_example
from qa_protocol.prompting import build_infer_prompt, build_teacher_prompt
from sft_trainer.formatting import build_sft_record


class FakePrefilterJudge:
    def __init__(self, decision: str, margin: float, decisions_by_answer: dict[str, tuple[str, float]] | None = None) -> None:
        self.decision = decision
        self.margin = margin
        self.decisions_by_answer = decisions_by_answer or {}
        self.rows = []

    def judge(self, rows):
        self.rows.extend(rows)
        return (
            [
                {
                    **row,
                    "judge": {
                        "margin": self.decisions_by_answer.get(str(row.get("answer") or ""), (self.decision, self.margin))[1],
                        "full_binary": {
                            "decision": self.decisions_by_answer.get(str(row.get("answer") or ""), (self.decision, self.margin))[0]
                        },
                    },
                }
                for row in rows
            ],
            {"num_rows": float(len(rows))},
        )


class FakeGenerator:
    def __init__(self, *args, **kwargs) -> None:
        self.prompts = []

    def generate_many(self, prompts):
        self.prompts.extend(prompts)
        return [json.dumps({"prompt_index": idx}) for idx, _ in enumerate(prompts)]


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _example() -> QaExample:
    return QaExample(
        id="s1:answerable",
        source_id="s1",
        variant_id="answerable",
        data_split="train_sft_raw",
        answerability_split="both",
        question="Who founded Acme?",
        knowledge="Doc1: Support sentence one.\n\nDoc2: Support sentence two.",
        reference_answer="Alice",
        answerability_label="answerable",
        difficulty="medium",
        supporting_sentences=(
            SupportingSentence(
                title="Doc1",
                sent_id=1,
                sentence="Support sentence one.",
                window_sentences=("Neighbor sentence.", "Support sentence one."),
                window_text="Doc1: Support sentence one.",
            ),
            SupportingSentence(
                title="Doc2",
                sent_id=0,
                sentence="Support sentence two.",
                window_sentences=("Support sentence two.",),
                window_text="Doc2: Support sentence two.",
            ),
        ),
        context_documents=(
            ContextDocument(title="Doc1", sentences=("Intro.", "Support sentence one.", "Neighbor sentence.")),
            ContextDocument(title="Doc2", sentences=("Support sentence two.", "Tail.")),
        ),
        metadata={},
    )


class TaggingTests(unittest.TestCase):
    def test_tagging_uses_two_independent_salted_assignments(self) -> None:
        source_id = "hotpot-123"
        data_split = assign_data_split_name(source_id, DataSplitConfig())
        answerability_split = assign_answerability_split_name(source_id, AnswerabilitySplitConfig())
        self.assertIn(data_split, {"validation", "test", "train_sft_raw", "train_dpo_raw"})
        self.assertIn(answerability_split, {"answerable", "unanswerable", "both"})
        self.assertNotEqual(
            assign_data_split_name(source_id, DataSplitConfig(hash_salt="data_split")),
            assign_data_split_name(source_id, DataSplitConfig(hash_salt="different_data_split")),
        )


class UnanswerableBuilderTests(unittest.TestCase):
    def test_build_unanswerable_replaces_support_with_neighbor_sentence(self) -> None:
        example = _example()
        built = build_unanswerable_from_answerable_example(
            example,
            UnanswerableBuildConfig(seed=7),
            origin_track="paired_answerable",
            raw_source_id=example.source_id,
            data_split=example.data_split,
            answerability_split=example.answerability_split,
        )
        self.assertIsNotNone(built)
        self.assertEqual(built.answerability_label, "unanswerable")
        self.assertNotEqual(built.knowledge, example.knowledge)
        self.assertEqual(built.data_split, "train_sft_raw")
        self.assertEqual(built.answerability_split, "both")
        self.assertIn("replacement_sentences", built.metadata["negative_strategy"])

    def test_build_examples_from_tagged_row_expands_both_to_two_examples(self) -> None:
        row = {
            "_id": "s2",
            "question": "Who founded Acme?",
            "answer": "Alice",
            "type": "bridge",
            "level": "medium",
            "data_split": "train_sft_raw",
            "answerability_split": "both",
            "context": {
                "title": ["Doc1", "Doc2"],
                "sentences": [["Intro.", "Support sentence one.", "Neighbor sentence."], ["Support sentence two.", "Tail."]],
            },
            "supporting_facts": {"title": ["Doc1", "Doc2"], "sent_id": [1, 0]},
        }
        built = build_examples_from_tagged_row(row, ConstructionConfig(), UnanswerableBuildConfig(seed=7))
        self.assertEqual(len(built), 2)
        self.assertEqual({item.answerability_label for item in built}, {"answerable", "unanswerable"})


class UnanswerablePrefilterTests(unittest.TestCase):
    def test_reference_answer_hypothesis_uses_fixed_template(self) -> None:
        self.assertEqual(build_reference_answer_hypothesis("Alpha"), "The answer is Alpha.")
        self.assertEqual(flip_yes_no_answer("yes"), "no")
        self.assertEqual(flip_yes_no_answer("No"), "yes")
        self.assertIsNone(flip_yes_no_answer("Alice"))

    def test_prefilter_mixed_examples_only_checks_unanswerable_rows(self) -> None:
        rows = [
            {
                "id": "a",
                "question": "Q1",
                "knowledge": "K1",
                "reference_answer": "A1",
                "answerability_label": "answerable",
                "metadata": {},
            },
            {
                "id": "u",
                "question": "Q2",
                "knowledge": "K2",
                "reference_answer": "A2",
                "answerability_label": "unanswerable",
                "metadata": {},
            },
        ]
        judge = FakePrefilterJudge(decision="no", margin=-0.4)
        with patch("qa_checks.source_prefilter.count_text_tokens", side_effect=lambda text, _: len(str(text).split())):
            kept, metrics = prefilter_mixed_examples(
                rows,
                tokenizer_name="dummy",
                max_prompt_tokens=512,
                enable_unanswerable_nli=True,
                judge=judge,
            )
        self.assertEqual(len(kept), 2)
        self.assertEqual(metrics["num_unanswerable_checked"], 1)
        self.assertEqual(len(judge.rows), 1)
        self.assertEqual(judge.rows[0]["question"], "Q2")

    def test_check_reference_answer_unsupported_keeps_only_full_binary_no(self) -> None:
        keep_report = check_reference_answer_unsupported(
            knowledge="K",
            question="Q",
            reference_answer="A",
            judge=FakePrefilterJudge(decision="no", margin=-0.5),
        )
        drop_report = check_reference_answer_unsupported(
            knowledge="K",
            question="Q",
            reference_answer="A",
            judge=FakePrefilterJudge(decision="yes", margin=0.5),
        )
        self.assertTrue(keep_report.keep)
        self.assertFalse(drop_report.keep)

    def test_yes_no_prefilter_requires_original_and_flipped_to_be_negative(self) -> None:
        judge = FakePrefilterJudge(
            decision="no",
            margin=-0.3,
            decisions_by_answer={
                "The answer is yes.": ("no", -0.6),
                "The answer is no.": ("yes", 0.2),
            },
        )
        report = check_reference_answer_unsupported(
            knowledge="K",
            question="Q",
            reference_answer="yes",
            judge=judge,
        )
        self.assertFalse(report.keep)
        self.assertTrue(report.used_flipped_check)
        self.assertEqual(report.full_binary_decision, "no")
        self.assertEqual(report.flipped_full_binary_decision, "yes")
        self.assertAlmostEqual(report.score, (-0.6 + 0.2) / 2.0)


class UnanswerablePromptTests(unittest.TestCase):
    def test_teacher_prompt_mentions_missing_or_ambiguous_rationale(self) -> None:
        prompt = build_teacher_prompt(knowledge="Knowledge", question="Question?", reference_answer=None)
        self.assertIn("missing, ambiguous, or contradictory", prompt)
        self.assertNotIn("Reference answer:", prompt)

    def test_infer_prompt_mentions_missing_or_ambiguous_rationale(self) -> None:
        prompt = build_infer_prompt(knowledge="Knowledge", question="Question?")
        self.assertIn("missing, ambiguous, or contradictory", prompt)


class MixedValidateTests(unittest.TestCase):
    @patch("grounded_qa.workflow.count_text_tokens", side_effect=lambda text, _: len(str(text).split()))
    def test_validate_mixed_routes_unanswerable_without_semantics(self, _mock_tokens) -> None:
        rows = [
            {
                "id": "u1",
                "source_id": "u1",
                "candidate_id": 1,
                "question": "Question?",
                "knowledge": "Knowledge.",
                "answerability_label": "unanswerable",
                "raw_output": json.dumps(
                    {
                        "answerability": "unanswerable",
                        "evidence": [],
                        "rationale": "The knowledge misses the needed fact.",
                        "answer": "I don't know based on the provided knowledge.",
                        "confidence": "medium",
                    }
                ),
            },
            {
                "id": "u1",
                "source_id": "u1",
                "candidate_id": 0,
                "question": "Question?",
                "knowledge": "Knowledge.",
                "answerability_label": "unanswerable",
                "raw_output": json.dumps(
                    {
                        "answerability": "unanswerable",
                        "evidence": [],
                        "rationale": "The knowledge does not identify the missing entity.",
                        "answer": "I don't know based on the provided knowledge.",
                        "confidence": "high",
                    }
                ),
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / "teacher.jsonl"
            out_path = Path(tmpdir) / "validated.jsonl"
            selected_path = Path(tmpdir) / "selected.jsonl"
            _write_jsonl(in_path, rows)

            validate_mixed_teacher_candidates(
                in_path=str(in_path),
                out_path=str(out_path),
                selected_out_path=str(selected_path),
                metrics_out=None,
                validation_cfg=ValidationConfig(enable_semantics=False, semantic_drop_by_nli=False),
            )

            selected = _read_jsonl(selected_path)
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["candidate_id"], 0)
            self.assertIsNone(selected[0]["validation_report"]["correctness"])
            self.assertIsNone(selected[0]["validation_report"]["semantics"])

    @patch("grounded_qa.workflow.count_text_tokens", side_effect=lambda text, _: len(str(text).split()))
    def test_validate_mixed_derives_unanswerable_confidence_from_prefilter_score(self, _mock_tokens) -> None:
        rows = [
            {
                "id": "u1",
                "source_id": "u1",
                "candidate_id": 0,
                "question": "Question 1?",
                "knowledge": "Knowledge 1.",
                "answerability_label": "unanswerable",
                "metadata": {"nli_prefilter": {"score": -0.8}},
                "raw_output": json.dumps(
                    {
                        "answerability": "unanswerable",
                        "evidence": [],
                        "rationale": "The key fact is missing.",
                        "answer": "I don't know based on the provided knowledge.",
                        "confidence": "low",
                    }
                ),
            },
            {
                "id": "u2",
                "source_id": "u2",
                "candidate_id": 0,
                "question": "Question 2?",
                "knowledge": "Knowledge 2.",
                "answerability_label": "unanswerable",
                "metadata": {"nli_prefilter": {"score": 0.1}},
                "raw_output": json.dumps(
                    {
                        "answerability": "unanswerable",
                        "evidence": [],
                        "rationale": "The key fact is missing.",
                        "answer": "I don't know based on the provided knowledge.",
                        "confidence": "low",
                    }
                ),
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / "teacher.jsonl"
            out_path = Path(tmpdir) / "validated.jsonl"
            selected_path = Path(tmpdir) / "selected.jsonl"
            _write_jsonl(in_path, rows)

            validate_mixed_teacher_candidates(
                in_path=str(in_path),
                out_path=str(out_path),
                selected_out_path=str(selected_path),
                metrics_out=None,
                validation_cfg=ValidationConfig(enable_semantics=False, semantic_drop_by_nli=False),
            )

            selected = _read_jsonl(selected_path)
            by_id = {row["id"]: row for row in selected}
            self.assertEqual(by_id["u1"]["parsed_output"]["confidence"], "high")
            self.assertEqual(by_id["u2"]["parsed_output"]["confidence"], "low")
            self.assertEqual(by_id["u1"]["validation_report"]["derived_confidence"], "high")
            self.assertEqual(by_id["u2"]["validation_report"]["derived_confidence"], "low")


class TeacherGenerateTests(unittest.TestCase):
    @patch("grounded_qa.workflow.OpenAICompatibleTextGenerator", return_value=FakeGenerator())
    def test_teacher_generate_uses_different_candidate_counts_by_answerability(self, _mock_generator) -> None:
        rows = [
            {
                "id": "a1",
                "source_id": "a1",
                "question": "Who founded Acme?",
                "knowledge": "Knowledge A",
                "reference_answer": "Alice",
                "answerability_label": "answerable",
            },
            {
                "id": "u1",
                "source_id": "u1",
                "question": "Who founded Beta?",
                "knowledge": "Knowledge U",
                "reference_answer": "Bob",
                "answerability_label": "unanswerable",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / "in.jsonl"
            out_path = Path(tmpdir) / "out.jsonl"
            _write_jsonl(in_path, rows)
            metrics = generate_teacher_candidates(
                in_path=str(in_path),
                out_path=str(out_path),
                metrics_out=None,
                cfg=TeacherGenerationConfig(
                    answerable_num_candidates_per_example=3,
                    unanswerable_num_candidates_per_example=1,
                ),
            )
            generated = _read_jsonl(out_path)
            self.assertEqual(metrics["num_candidates"], 4)
            self.assertEqual(len(generated), 4)
            self.assertEqual(sum(1 for row in generated if row["answerability_label"] == "answerable"), 3)
            self.assertEqual(sum(1 for row in generated if row["answerability_label"] == "unanswerable"), 1)


class MixedSftPackingTests(unittest.TestCase):
    def test_unanswerable_sft_record_keeps_teacher_confidence(self) -> None:
        row = {
            "id": "u2",
            "source_id": "u2",
            "data_split": "train_sft_raw",
            "knowledge": "Knowledge.",
            "question": "Question?",
            "answerability_split": "unanswerable",
            "parsed_output": {
                "answerability": "unanswerable",
                "evidence": [],
                "rationale": "The key fact is missing from the knowledge.",
                "answer": "I don't know based on the provided knowledge.",
                "confidence": "low",
            },
            "validation_report": {
                "overall_ok": True,
                "derived_confidence": None,
            },
        }

        record = build_sft_record(row=row, prompt_style="infer_v1")
        self.assertIsNotNone(record)
        self.assertEqual(record["data_split"], "train_sft_raw")
        self.assertEqual(record["target_structured"]["confidence"], "low")


if __name__ == "__main__":
    unittest.main()
