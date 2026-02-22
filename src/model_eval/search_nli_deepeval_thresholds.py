from __future__ import annotations

import argparse
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from qa_judge.config import NLIConfig
    from qa_judge.nli import NLIVerifier
    from prompt import build_qa_premise
except ImportError:  # pragma: no cover - compatibility fallback for editable installs.
    from qa_judge.config import NLIConfig
    from qa_judge.nli import NLIVerifier
    from prompt import build_qa_premise

from dataio import write_jsonl
from .common import group_rows_by_model, write_csv
from .eval_nli_deepeval_consistency import _binary_stats, _load_deepeval_rows, _mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Input: DeepEval details artifact.
    parser.add_argument(
        "--deepeval_details_path",
        type=str,
        required=True,
        help="DeepEval details path (JSONL or save_to_disk dataset).",
    )
    parser.add_argument("--split", type=str, default="train", help="Split name when details path is a DatasetDict.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum rows to compare. -1 means all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_error_rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip rows where DeepEval reported non-empty error.",
    )
    parser.add_argument(
        "--deepeval_success_means_supported",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True, DeepEval success=True is treated as faithful/supported.",
    )

    # NLI scorer setup.
    parser.add_argument("--nli_model_name", type=str, default=NLIConfig.model_name)
    parser.add_argument("--secondary_nli_model_name", type=str, default=None)
    parser.add_argument("--nli_device", type=str, default=NLIConfig.device)
    parser.add_argument("--nli_batch_size", type=int, default=NLIConfig.batch_size)
    parser.add_argument("--nli_max_length", type=int, default=NLIConfig.max_length)
    parser.add_argument(
        "--nli_fp16",
        action=argparse.BooleanOptionalAction,
        default=NLIConfig.fp16,
        help="Whether to enable fp16 for NLI scoring on CUDA.",
    )
    parser.add_argument("--vote_mode", type=str, default="primary", choices=["primary", "and", "or"])

    # Threshold search space.
    parser.add_argument(
        "--candidate_entail_values",
        type=str,
        default="",
        help="Comma-separated entail thresholds. If set, overrides entail min/max/step.",
    )
    parser.add_argument("--candidate_entail_min", type=float, default=0.40)
    parser.add_argument("--candidate_entail_max", type=float, default=0.90)
    parser.add_argument("--candidate_entail_step", type=float, default=0.05)

    parser.add_argument(
        "--candidate_contradict_values",
        type=str,
        default="",
        help="Comma-separated contradict thresholds. If set, overrides contradict min/max/step.",
    )
    parser.add_argument("--candidate_contradict_min", type=float, default=0.05)
    parser.add_argument("--candidate_contradict_max", type=float, default=0.40)
    parser.add_argument("--candidate_contradict_step", type=float, default=0.05)

    # Optimization objective.
    parser.add_argument(
        "--objective",
        type=str,
        default="cohen_kappa",
        choices=["cohen_kappa", "faithful_f1"],
        help="Metric to maximize during threshold search.",
    )

    # Outputs.
    parser.add_argument("--out_csv", type=str, default="nli_deepeval_threshold_search.csv")
    parser.add_argument("--out_jsonl", type=str, default="nli_deepeval_threshold_search_best.jsonl")
    return parser.parse_args()


def _parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    for item in text.split(","):
        token = item.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def _build_values(explicit: str, v_min: float, v_max: float, v_step: float, name: str) -> List[float]:
    if explicit.strip():
        values = sorted(set(_parse_float_list(explicit)))
        if not values:
            raise ValueError(f"{name} explicit values are empty.")
        return values

    if v_step <= 0.0:
        raise ValueError(f"{name} step must be > 0, got {v_step}")
    if v_max < v_min:
        raise ValueError(f"{name} max must be >= min, got min={v_min}, max={v_max}")

    values: List[float] = []
    cur = v_min
    while cur <= v_max + 1e-12:
        values.append(round(cur, 10))
        cur += v_step
    return values


def _evaluate_thresholds(
    *,
    rows: Sequence[Dict[str, Any]],
    entail_threshold: float,
    contradict_threshold: float,
    vote_mode: str,
) -> Dict[str, Any]:
    preds: List[bool] = []
    refs: List[bool] = []
    nli_supported_values: List[float] = []

    for row in rows:
        p_entail = float(row["nli_candidate_entail_primary"])
        p_contra = float(row["nli_candidate_contradict_primary"])
        p_supported = p_entail >= entail_threshold and p_contra <= contradict_threshold

        s_supported: Optional[bool] = None
        if row.get("nli_candidate_entail_secondary") is not None and row.get("nli_candidate_contradict_secondary") is not None:
            s_entail = float(row["nli_candidate_entail_secondary"])
            s_contra = float(row["nli_candidate_contradict_secondary"])
            s_supported = s_entail >= entail_threshold and s_contra <= contradict_threshold

        if vote_mode == "primary" or s_supported is None:
            supported = p_supported
        elif vote_mode == "and":
            supported = bool(p_supported and s_supported)
        else:
            supported = bool(p_supported or s_supported)

        nli_supported_values.append(1.0 if supported else 0.0)

        deepeval_success = row.get("deepeval_success")
        if deepeval_success is None:
            continue
        preds.append(supported)
        refs.append(bool(deepeval_success))

    bstats = _binary_stats(preds, refs)
    return {
        "num_rows": len(rows),
        "num_comparable": len(preds),
        "deepeval_pass_rate": _mean([1.0 if bool(x) else 0.0 for x in refs]),
        "nli_supported_rate": _mean(nli_supported_values),
        "nli_accuracy_vs_deepeval": bstats["accuracy"],
        "agreement_rate": bstats["agreement_rate"],
        "cohen_kappa": bstats["cohen_kappa"],
        "faithful_precision": bstats["precision"],
        "faithful_recall": bstats["recall"],
        "faithful_f1": bstats["f1"],
        "tp": bstats["tp"],
        "tn": bstats["tn"],
        "fp": bstats["fp"],
        "fn": bstats["fn"],
    }


def _score_for_sort(value: Any) -> float:
    if value is None:
        return -float("inf")
    try:
        f = float(value)
    except (TypeError, ValueError):
        return -float("inf")
    if math.isnan(f):
        return -float("inf")
    return f


def _choose_best(rows: Sequence[Dict[str, Any]], objective: str) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[float, float, float]] = None

    for row in rows:
        key = (
            _score_for_sort(row.get(objective)),
            _score_for_sort(row.get("cohen_kappa")),
            _score_for_sort(row.get("faithful_f1")),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = row
    return best


def _score_rows_with_nli(
    *,
    rows: Sequence[Dict[str, Any]],
    primary: NLIVerifier,
    secondary: Optional[NLIVerifier],
) -> List[Dict[str, Any]]:
    premises = [build_qa_premise(str(row["knowledge"]), str(row["question"])) for row in rows]
    hypotheses = [str(row["answer"]) for row in rows]

    primary_scores = primary.score(premises, hypotheses)
    secondary_scores = secondary.score(premises, hypotheses) if secondary is not None else None

    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        p_sc = primary_scores[i]
        s_sc = None if secondary_scores is None else secondary_scores[i]
        out.append(
            {
                **row,
                "nli_candidate_entail_primary": float(p_sc.entail),
                "nli_candidate_contradict_primary": float(p_sc.contradict),
                "nli_candidate_entail_secondary": None if s_sc is None else float(s_sc.entail),
                "nli_candidate_contradict_secondary": None if s_sc is None else float(s_sc.contradict),
            }
        )
    return out


def _iter_grid(entail_values: Iterable[float], contradict_values: Iterable[float]) -> Iterable[Tuple[float, float]]:
    for entail in entail_values:
        for contradict in contradict_values:
            yield float(entail), float(contradict)


def main() -> None:
    args = parse_args()

    entail_values = _build_values(
        explicit=args.candidate_entail_values,
        v_min=args.candidate_entail_min,
        v_max=args.candidate_entail_max,
        v_step=args.candidate_entail_step,
        name="candidate_entail",
    )
    contradict_values = _build_values(
        explicit=args.candidate_contradict_values,
        v_min=args.candidate_contradict_min,
        v_max=args.candidate_contradict_max,
        v_step=args.candidate_contradict_step,
        name="candidate_contradict",
    )

    base_rows = _load_deepeval_rows(args)
    if not base_rows:
        raise RuntimeError("No valid DeepEval detail rows found for threshold search.")

    primary = NLIVerifier(
        model_name=args.nli_model_name,
        device=args.nli_device,
        batch_size=args.nli_batch_size,
        max_length=args.nli_max_length,
        fp16=args.nli_fp16,
    )
    secondary = None
    if args.secondary_nli_model_name:
        secondary = NLIVerifier(
            model_name=args.secondary_nli_model_name,
            device=args.nli_device,
            batch_size=args.nli_batch_size,
            max_length=args.nli_max_length,
            fp16=args.nli_fp16,
        )

    grouped = group_rows_by_model(base_rows)
    grid_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []

    for _, model_rows in sorted(grouped.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        scored_rows = _score_rows_with_nli(rows=model_rows, primary=primary, secondary=secondary)

        one_model_rows: List[Dict[str, Any]] = []
        for entail_th, contra_th in _iter_grid(entail_values, contradict_values):
            metrics = _evaluate_thresholds(
                rows=scored_rows,
                entail_threshold=entail_th,
                contradict_threshold=contra_th,
                vote_mode=args.vote_mode,
            )
            row = {
                "model_tag": str(model_rows[0]["model_tag"]),
                "model_step": int(model_rows[0]["model_step"]),
                "model_path": str(model_rows[0]["model_path"]),
                "candidate_entail_threshold": entail_th,
                "candidate_contradict_threshold": contra_th,
                "objective": args.objective,
                "objective_value": metrics[args.objective],
                **metrics,
            }
            one_model_rows.append(row)
            grid_rows.append(row)

        best = _choose_best(one_model_rows, args.objective)
        if best is not None:
            best_rows.append(
                {
                    **best,
                    "search_num_trials": len(one_model_rows),
                    "search_entail_values": ",".join(str(x) for x in entail_values),
                    "search_contradict_values": ",".join(str(x) for x in contradict_values),
                }
            )

    best_keys = {
        (str(r["model_tag"]), int(r["model_step"]), str(r["model_path"]), float(r["candidate_entail_threshold"]), float(r["candidate_contradict_threshold"]))
        for r in best_rows
    }
    for row in grid_rows:
        key = (
            str(row["model_tag"]),
            int(row["model_step"]),
            str(row["model_path"]),
            float(row["candidate_entail_threshold"]),
            float(row["candidate_contradict_threshold"]),
        )
        row["is_best"] = key in best_keys

    write_csv(grid_rows, args.out_csv)
    print(f"Saved search grid: {args.out_csv}")

    write_jsonl(args.out_jsonl, best_rows)
    print(f"Saved best thresholds: {args.out_jsonl}")


if __name__ == "__main__":
    main()
