from __future__ import annotations

import argparse

from dataio import write_json
from project_config.resolve import resolve_eval_args

from .common import write_csv
from .summary_builders import build_validation_summary_rows, read_csv_rows, select_best_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_curve_path", type=str, required=True, help="Validation structured evaluation CSV.")
    parser.add_argument("--loss_curve_path", type=str, required=True, help="Validation SFT loss CSV.")
    parser.add_argument("--out_csv", type=str, required=True, help="Validation summary CSV output path.")
    parser.add_argument("--selection_out", type=str, required=True, help="Selection JSON output path.")
    parser.add_argument("--parse_ok_threshold", type=float, default=None)
    parser.add_argument("--protocol_ok_threshold", type=float, default=None)
    parser.add_argument("--evidence_ok_threshold", type=float, default=None)
    return resolve_eval_args(parser.parse_args(), preset="selection")


def run_build_validation_summary(args: argparse.Namespace) -> tuple[list[dict], dict]:
    eval_rows = read_csv_rows(args.eval_curve_path)
    loss_rows = read_csv_rows(args.loss_curve_path)
    selection = select_best_checkpoint(
        eval_rows=eval_rows,
        loss_rows=loss_rows,
        parse_ok_threshold=args.parse_ok_threshold,
        protocol_ok_threshold=args.protocol_ok_threshold,
        evidence_ok_threshold=args.evidence_ok_threshold,
    )
    summary_rows = build_validation_summary_rows(
        eval_rows=eval_rows,
        loss_rows=loss_rows,
        selection=selection,
        parse_ok_threshold=args.parse_ok_threshold,
        protocol_ok_threshold=args.protocol_ok_threshold,
        evidence_ok_threshold=args.evidence_ok_threshold,
    )
    return summary_rows, selection


def main() -> None:
    args = parse_args()
    summary_rows, selection = run_build_validation_summary(args)
    write_csv(summary_rows, args.out_csv)
    write_json(args.selection_out, selection)
    print(f"Saved validation summary to: {args.out_csv}")
    print(f"Saved checkpoint selection to: {args.selection_out}")


if __name__ == "__main__":
    main()
