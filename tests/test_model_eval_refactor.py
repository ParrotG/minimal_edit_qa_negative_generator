from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from model_eval.common import (
        extract_knowledge_question_from_infer_prompt,
        load_dataset_split,
        load_structured_generation_items,
        normalize_generated_row,
    )
except ModuleNotFoundError:  # pragma: no cover
    extract_knowledge_question_from_infer_prompt = None
    load_dataset_split = None
    load_structured_generation_items = None
    normalize_generated_row = None


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@unittest.skipUnless(
    all(
        item is not None
        for item in (
            extract_knowledge_question_from_infer_prompt,
            load_dataset_split,
            load_structured_generation_items,
            normalize_generated_row,
        )
    ),
    "model_eval dependencies are not available",
)
class ModelEvalRefactorTests(unittest.TestCase):
    def test_extract_knowledge_question_from_infer_prompt(self) -> None:
        prompt = (
            "Return exactly one JSON object.\n\n"
            "Knowledge:\nDocA: Alpha is true.\n\nDocB: Beta is true.\n\n"
            "Question: Who is true?\n"
        )
        knowledge, question = extract_knowledge_question_from_infer_prompt(prompt)
        self.assertIn("DocA: Alpha is true.", knowledge)
        self.assertEqual(question, "Who is true?")

    def test_normalize_generated_row_supports_rationale_plus_answer(self) -> None:
        row = {
            "model_tag": "base",
            "model_step": 0,
            "model_path": "Qwen/Qwen3-0.6B",
            "source_id": "s1",
            "question": "Who founded Acme?",
            "knowledge": "Acme was founded by Alice.",
            "parsed_output": {
                "answerability": "answerable",
                "evidence": [{"quote": "Acme was founded by Alice."}],
                "rationale": "The knowledge explicitly names the founder as Alice.",
                "answer": "Alice",
                "confidence": "high",
            },
        }
        normalized = normalize_generated_row(row, idx=0, answer_source="rationale_plus_answer")
        self.assertIsNotNone(normalized)
        self.assertIn("Therefore the answer is Alice", normalized["answer"])

    def test_load_dataset_split_handles_jsonl_with_mixed_nested_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mixed.jsonl"
            _write_jsonl(
                path,
                [
                    {
                        "id": "a",
                        "prompt": "Knowledge:\nK1\n\nQuestion: Q1",
                        "completion": '{"answer":"A1"}',
                        "metadata": {"source_prefilter": {"infer_prompt_tokens": 128}},
                    },
                    {
                        "id": "b",
                        "prompt": "Knowledge:\nK2\n\nQuestion: Q2",
                        "completion": '{"answer":"A2"}',
                        "metadata": {"source_prefilter": {"infer_prompt_tokens": 64, "issues": []}, "nli_prefilter": {"keep": True}},
                    },
                ],
            )
            ds = load_dataset_split(str(path), split="train")
            self.assertEqual(len(ds), 2)

    def test_load_structured_generation_items_from_prompt_only_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            _write_jsonl(
                path,
                [
                    {
                        "id": "x1",
                        "source_id": "s1",
                        "prompt": "Instruction\n\nKnowledge:\nDoc1: Alice founded Acme.\n\nQuestion: Who founded Acme?\n",
                        "metadata": {"reference_answer": "Alice", "answerability_label": "answerable"},
                    }
                ],
            )
            items = load_structured_generation_items(
                data_path=str(path),
                split="train",
                max_samples=-1,
                seed=42,
            )
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["question"], "Who founded Acme?")
            self.assertIn("Alice founded Acme", items[0]["knowledge"])
            self.assertEqual(items[0]["reference_answer"], "Alice")


if __name__ == "__main__":
    unittest.main()
