from __future__ import annotations

import argparse

from .common import write_csv
from .summary_builders import build_test_summary_rows, load_jsonl_rows, read_csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curve_path",
        type=str,
        action="append",
        required=True,
        help="Repeat for each test track evaluation summary CSV.",
    )
    parser.add_argument(
        "--details_path",
        type=str,
        action="append",
        required=True,
        help="Repeat for each test track evaluation details JSONL.",
    )
    parser.add_argument(
        "--deepeval_details_path",
        type=str,
        action="append",
        required=True,
        help="Repeat for each test track DeepEval details JSONL.",
    )
    parser.add_argument("--out_csv", type=str, required=True, help="Test summary CSV output path.")
    return parser.parse_args()


def run_build_test_summary(args: argparse.Namespace) -> list[dict]:
    curve_rows: list[dict] = []
    for path in args.curve_path:
        curve_rows.extend(read_csv_rows(path))
    detail_rows = load_jsonl_rows(args.details_path)
    deepeval_detail_rows = load_jsonl_rows(args.deepeval_details_path)
    return build_test_summary_rows(
        track_rows=curve_rows,
        detail_rows=detail_rows,
        deepeval_detail_rows=deepeval_detail_rows,
    )


def main() -> None:
    args = parse_args()
    summary_rows = run_build_test_summary(args)
    write_csv(summary_rows, args.out_csv)
    print(f"Saved test summary to: {args.out_csv}")


if __name__ == "__main__":
    main()
