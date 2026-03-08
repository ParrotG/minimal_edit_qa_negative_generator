from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from datasets import load_from_disk
except ModuleNotFoundError:  # pragma: no cover
    load_from_disk = None

try:
    from sft_trainer import prepare_sft_dataset as prep
except ModuleNotFoundError:  # pragma: no cover
    prep = None


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@unittest.skipUnless(load_from_disk is not None and prep is not None, "datasets package is not available")
class PrepareSftDatasetTests(unittest.TestCase):
    def test_prepare_dataset_supports_multi_source_caps_and_token_refilter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            train_path = tmp / "train.jsonl"
            validation_path = tmp / "validation.jsonl"
            test_path = tmp / "test.jsonl"
            out_dir = tmp / "dataset_out"
            metrics_out = tmp / "metrics.json"

            _write_jsonl(
                train_path,
                [
                    {
                        "id": "tr-a-1",
                        "source_id": "s1",
                        "prompt": "short prompt",
                        "completion": '{"answer":"A"}',
                        "answerability_label": "answerable",
                    },
                    {
                        "id": "tr-a-2",
                        "source_id": "s2",
                        "prompt": "another short prompt",
                        "completion": '{"answer":"B"}',
                        "answerability_label": "answerable",
                    },
                    {
                        "id": "tr-u-drop",
                        "source_id": "s3",
                        "prompt": (
                            "this prompt is intentionally too long for the configured token filter budget "
                            "in this unit test and keeps adding extra filler tokens so it exceeds the limit"
                        ),
                        "completion": '{"answer":"C"}',
                        "answerability_label": "unanswerable",
                    },
                    {
                        "id": "tr-u-1",
                        "source_id": "s4",
                        "prompt": "tiny prompt",
                        "completion": '{"answer":"D"}',
                        "answerability_label": "unanswerable",
                    },
                ],
            )
            _write_jsonl(
                validation_path,
                [
                    {
                        "id": "va-1",
                        "source_id": "s5",
                        "prompt": "valid prompt",
                        "completion": '{"answer":"V"}',
                        "metadata": {"answerability_label": "answerable"},
                    }
                ],
            )
            _write_jsonl(
                test_path,
                [
                    {
                        "id": "te-1",
                        "source_id": "s6",
                        "prompt": "test prompt",
                        "question": "Who founded Acme?",
                        "knowledge": "Acme was founded by Alice.",
                        "answerability_label": "answerable",
                    },
                    {
                        "id": "te-drop",
                        "source_id": "s7",
                        "knowledge": "Missing question, should be dropped.",
                        "answerability_label": "unanswerable",
                    },
                ],
            )

            def fake_count_text_tokens_batch(texts, tokenizer_name, batch_size=128):
                return [len(str(text).split()) for text in texts]

            argv = [
                "prepare_sft_dataset.py",
                "--train_paths",
                str(train_path),
                "--validation_paths",
                str(validation_path),
                "--test_paths",
                str(test_path),
                "--output_dir",
                str(out_dir),
                "--metrics_out",
                str(metrics_out),
                "--max_prompt_tokens",
                "16",
                "--max_completion_tokens",
                "20",
                "--max_train_answerable_samples",
                "1",
                "--max_train_unanswerable_samples",
                "1",
                "--max_train_samples",
                "2",
                "--max_test_samples",
                "1",
                "--no_shuffle",
            ]
            with patch.object(prep, "count_text_tokens_batch", side_effect=fake_count_text_tokens_batch):
                with patch("sys.argv", argv):
                    prep.main()

            ds = load_from_disk(str(out_dir))
            self.assertEqual(set(ds.keys()), {"train", "validation", "test"})
            self.assertEqual(len(ds["train"]), 2)
            self.assertEqual(len(ds["validation"]), 1)
            self.assertEqual(len(ds["test"]), 1)

            train_ids = set(ds["train"]["id"])
            self.assertEqual(train_ids, {"tr-a-1", "tr-u-1"})

            test_completion = ds["test"]["completion"][0]
            self.assertEqual(test_completion, "")
            self.assertTrue(ds["test"]["prompt"][0].strip())

            metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
            self.assertEqual(metrics["splits"]["train"], 2)
            self.assertEqual(metrics["sampling"]["train"]["num_answerable_kept"], 1)
            self.assertEqual(metrics["sampling"]["train"]["num_unanswerable_kept"], 1)
            self.assertEqual(metrics["token_filter"]["train"]["num_drop_prompt_over_budget"], 1)


if __name__ == "__main__":
    unittest.main()
