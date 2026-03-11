from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List, Optional, Sequence

from project_config import PROJECT_SETTINGS

try:
    from src.dataio import write_jsonl
except ImportError:  # pragma: no cover - compatibility fallback for editable installs.
    from dataio import write_jsonl

from .common import binary_stats, build_values, group_rows_by_model, load_dataset_split, mean, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--scored_path", type=str, required=True, help="Input from score_nli_for_calibration.py.")
    parser.add_argument("--split", type=str, default="train", help="Split name when scored_path is a DatasetDict.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum rows to evaluate. -1 means all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_field", type=str, default="gold_supported", help="Boolean gold label field.")

    parser.add_argument(
        "--search_objective",
        type=str,
        default=PROJECT_SETTINGS.calibration.search_objective,
        choices=["cohen_kappa", "f1", "accuracy"],
        help="Objective for non-reject methods.",
    )
    parser.add_argument(
        "--reject_alpha",
        type=float,
        default=PROJECT_SETTINGS.calibration.reject_alpha,
        help="Lagrangian alpha in Error(kept) + alpha * RejectRate for reject methods.",
    )

    parser.add_argument("--margin_threshold_values", type=str, default="")
    parser.add_argument("--margin_threshold_min", type=float, default=PROJECT_SETTINGS.calibration.margin_threshold_min)
    parser.add_argument("--margin_threshold_max", type=float, default=PROJECT_SETTINGS.calibration.margin_threshold_max)
    parser.add_argument("--margin_threshold_step", type=float, default=PROJECT_SETTINGS.calibration.margin_threshold_step)

    parser.add_argument("--argmax_conf_threshold_values", type=str, default="")
    parser.add_argument("--argmax_conf_threshold_min", type=float, default=PROJECT_SETTINGS.calibration.argmax_conf_threshold_min)
    parser.add_argument("--argmax_conf_threshold_max", type=float, default=PROJECT_SETTINGS.calibration.argmax_conf_threshold_max)
    parser.add_argument("--argmax_conf_threshold_step", type=float, default=PROJECT_SETTINGS.calibration.argmax_conf_threshold_step)

    parser.add_argument("--band_half_width_values", type=str, default="")
    parser.add_argument("--band_half_width_min", type=float, default=PROJECT_SETTINGS.calibration.band_half_width_min)
    parser.add_argument("--band_half_width_max", type=float, default=PROJECT_SETTINGS.calibration.band_half_width_max)
    parser.add_argument("--band_half_width_step", type=float, default=PROJECT_SETTINGS.calibration.band_half_width_step)

    parser.add_argument("--out_csv", type=str, default="calibration_strategy_search.csv")
    parser.add_argument("--out_jsonl", type=str, default="calibration_strategy_search_best.jsonl")
    return parser.parse_args()


def _safe_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_scored_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    ds = load_dataset_split(data_path=args.scored_path, split=args.split).shuffle(seed=args.seed)
    out: List[Dict[str, Any]] = []

    for idx, row in enumerate(ds):
        entail = _safe_float(row.get("nli_entail"))
        neutral = _safe_float(row.get("nli_neutral"))
        contradict = _safe_float(row.get("nli_contradict"))
        margin = _safe_float(row.get("nli_margin_logit_entail_vs_max_other"))
        argmax_prob = _safe_float(row.get("nli_argmax_prob"))
        argmax_label = str(row.get("nli_argmax_label") or "")
        if entail is None or neutral is None or contradict is None:
            continue
        if margin is None:
            eps = 1e-12
            logit_ent = math.log(min(max(entail, eps), 1.0 - eps) / (1.0 - min(max(entail, eps), 1.0 - eps)))
            logit_neu = math.log(min(max(neutral, eps), 1.0 - eps) / (1.0 - min(max(neutral, eps), 1.0 - eps)))
            logit_con = math.log(min(max(contradict, eps), 1.0 - eps) / (1.0 - min(max(contradict, eps), 1.0 - eps)))
            margin = logit_ent - max(logit_neu, logit_con)
        if argmax_prob is None:
            argmax_prob = max(entail, neutral, contradict)
        if not argmax_label:
            if entail >= neutral and entail >= contradict:
                argmax_label = "entail"
            elif neutral >= entail and neutral >= contradict:
                argmax_label = "neutral"
            else:
                argmax_label = "contradict"

        out.append(
            {
                "model_tag": str(row.get("model_tag") or "model"),
                "model_step": int(row.get("model_step", 0)),
                "model_path": str(row.get("model_path") or row.get("model_tag") or "model"),
                "sample_id": int(row.get("sample_id", idx)),
                "source_id": str(row.get("source_id") or row.get("sample_id") or idx),
                "gold_supported": _safe_bool(row.get(args.label_field)),
                "nli_entail": entail,
                "nli_neutral": neutral,
                "nli_contradict": contradict,
                "nli_argmax_label": argmax_label,
                "nli_argmax_prob": float(argmax_prob),
                "nli_margin_logit_entail_vs_max_other": float(margin),
            }
        )
        if args.max_samples > 0 and len(out) >= args.max_samples:
            break
    return out


def _evaluate_predictions(
    *,
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[Optional[bool]],
    reject_alpha: float,
) -> Dict[str, float]:
    preds: List[bool] = []
    refs: List[bool] = []
    total_labeled = 0
    kept = 0

    for row, pred in zip(rows, predictions):
        label = row.get("gold_supported")
        if label is None:
            continue
        total_labeled += 1
        if pred is None:
            continue
        kept += 1
        preds.append(bool(pred))
        refs.append(bool(label))

    rejected = total_labeled - kept
    coverage = kept / total_labeled if total_labeled > 0 else float("nan")
    reject_rate = rejected / total_labeled if total_labeled > 0 else float("nan")
    metrics = binary_stats(preds, refs)
    error_kept = 1.0 - metrics["accuracy"] if kept > 0 else 1.0
    lagrangian = error_kept + reject_alpha * reject_rate if total_labeled > 0 else float("nan")

    return {
        "num_rows": float(len(rows)),
        "num_labeled": float(total_labeled),
        "num_kept": float(kept),
        "num_rejected": float(rejected),
        "coverage": float(coverage),
        "reject_rate": float(reject_rate),
        "error_kept": float(error_kept),
        "lagrangian_objective": float(lagrangian),
        "accuracy": metrics["accuracy"],
        "cohen_kappa": metrics["cohen_kappa"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "tp": metrics["tp"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
    }


def _method_argmax(rows: Sequence[Dict[str, Any]]) -> List[Optional[bool]]:
    return [row["nli_argmax_label"] == "entail" for row in rows]


def _method_margin(rows: Sequence[Dict[str, Any]], threshold: float) -> List[Optional[bool]]:
    return [float(row["nli_margin_logit_entail_vs_max_other"]) >= threshold for row in rows]


def _method_argmax_with_reject(rows: Sequence[Dict[str, Any]], min_conf: float) -> List[Optional[bool]]:
    out: List[Optional[bool]] = []
    base = _method_argmax(rows)
    for row, pred in zip(rows, base):
        out.append(pred if float(row["nli_argmax_prob"]) >= min_conf else None)
    return out


def _method_margin_with_reject_band(rows: Sequence[Dict[str, Any]], threshold: float, half_width: float) -> List[Optional[bool]]:
    out: List[Optional[bool]] = []
    for row in rows:
        margin = float(row["nli_margin_logit_entail_vs_max_other"])
        if abs(margin - threshold) <= half_width:
            out.append(None)
        else:
            out.append(margin >= threshold)
    return out


def _score_for_sort(value: Any) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return -float("inf")
    if math.isnan(num):
        return -float("inf")
    return num


def _score_for_sort_min(value: Any) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return float("inf")
    if math.isnan(num):
        return float("inf")
    return num


def _select_best(rows: Sequence[Dict[str, Any]], objective: str, minimize: bool) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    if minimize:
        return min(rows, key=lambda r: (_score_for_sort_min(r.get(objective)), -_score_for_sort(r.get("coverage"))))
    return max(rows, key=lambda r: (_score_for_sort(r.get(objective)), _score_for_sort(r.get("cohen_kappa"))))


def main() -> None:
    args = parse_args()
    rows = _load_scored_rows(args)
    if not rows:
        raise RuntimeError("No valid rows found from scored input.")

    margin_thresholds = build_values(
        explicit=args.margin_threshold_values,
        v_min=args.margin_threshold_min,
        v_max=args.margin_threshold_max,
        v_step=args.margin_threshold_step,
        name="margin_threshold",
    )
    argmax_conf_thresholds = build_values(
        explicit=args.argmax_conf_threshold_values,
        v_min=args.argmax_conf_threshold_min,
        v_max=args.argmax_conf_threshold_max,
        v_step=args.argmax_conf_threshold_step,
        name="argmax_conf_threshold",
    )
    band_half_widths = build_values(
        explicit=args.band_half_width_values,
        v_min=args.band_half_width_min,
        v_max=args.band_half_width_max,
        v_step=args.band_half_width_step,
        name="band_half_width",
    )

    grouped = group_rows_by_model(rows)
    trial_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []

    for _, model_rows in sorted(grouped.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        model_tag = str(model_rows[0]["model_tag"])
        model_step = int(model_rows[0]["model_step"])
        model_path = str(model_rows[0]["model_path"])

        method_rows: Dict[str, List[Dict[str, Any]]] = {
            "argmax_entail": [],
            "logit_margin_threshold": [],
            "argmax_with_conf_reject": [],
            "margin_with_reject_band": [],
        }

        argmax_preds = _method_argmax(model_rows)
        argmax_metrics = _evaluate_predictions(rows=model_rows, predictions=argmax_preds, reject_alpha=args.reject_alpha)
        argmax_row = {
            "model_tag": model_tag,
            "model_step": model_step,
            "model_path": model_path,
            "method": "argmax_entail",
            "margin_threshold": "",
            "argmax_conf_threshold": "",
            "band_half_width": "",
            "search_objective": args.search_objective,
            "search_objective_value": argmax_metrics[args.search_objective],
            "reject_alpha": args.reject_alpha,
            **argmax_metrics,
        }
        method_rows["argmax_entail"].append(argmax_row)
        trial_rows.append(argmax_row)

        for margin_th in margin_thresholds:
            preds = _method_margin(model_rows, threshold=margin_th)
            metrics = _evaluate_predictions(rows=model_rows, predictions=preds, reject_alpha=args.reject_alpha)
            row = {
                "model_tag": model_tag,
                "model_step": model_step,
                "model_path": model_path,
                "method": "logit_margin_threshold",
                "margin_threshold": margin_th,
                "argmax_conf_threshold": "",
                "band_half_width": "",
                "search_objective": args.search_objective,
                "search_objective_value": metrics[args.search_objective],
                "reject_alpha": args.reject_alpha,
                **metrics,
            }
            method_rows["logit_margin_threshold"].append(row)
            trial_rows.append(row)

        for conf_th in argmax_conf_thresholds:
            preds = _method_argmax_with_reject(model_rows, min_conf=conf_th)
            metrics = _evaluate_predictions(rows=model_rows, predictions=preds, reject_alpha=args.reject_alpha)
            row = {
                "model_tag": model_tag,
                "model_step": model_step,
                "model_path": model_path,
                "method": "argmax_with_conf_reject",
                "margin_threshold": "",
                "argmax_conf_threshold": conf_th,
                "band_half_width": "",
                "search_objective": args.search_objective,
                "search_objective_value": metrics[args.search_objective],
                "reject_alpha": args.reject_alpha,
                **metrics,
            }
            method_rows["argmax_with_conf_reject"].append(row)
            trial_rows.append(row)

        for margin_th in margin_thresholds:
            for half_width in band_half_widths:
                preds = _method_margin_with_reject_band(model_rows, threshold=margin_th, half_width=half_width)
                metrics = _evaluate_predictions(rows=model_rows, predictions=preds, reject_alpha=args.reject_alpha)
                row = {
                    "model_tag": model_tag,
                    "model_step": model_step,
                    "model_path": model_path,
                    "method": "margin_with_reject_band",
                    "margin_threshold": margin_th,
                    "argmax_conf_threshold": "",
                    "band_half_width": half_width,
                    "search_objective": args.search_objective,
                    "search_objective_value": metrics[args.search_objective],
                    "reject_alpha": args.reject_alpha,
                    **metrics,
                }
                method_rows["margin_with_reject_band"].append(row)
                trial_rows.append(row)

        for method_name, candidates in method_rows.items():
            if method_name in {"argmax_with_conf_reject", "margin_with_reject_band"}:
                best = _select_best(candidates, objective="lagrangian_objective", minimize=True)
            else:
                best = _select_best(candidates, objective=args.search_objective, minimize=False)
            if best is None:
                continue
            best_rows.append(
                {
                    **best,
                    "num_trials_method": len(candidates),
                    "used_margin_threshold_values": ",".join(str(x) for x in margin_thresholds),
                    "used_argmax_conf_threshold_values": ",".join(str(x) for x in argmax_conf_thresholds),
                    "used_band_half_width_values": ",".join(str(x) for x in band_half_widths),
                }
            )

    best_keys = {
        (
            str(r["model_tag"]),
            int(r["model_step"]),
            str(r["model_path"]),
            str(r["method"]),
            str(r["margin_threshold"]),
            str(r["argmax_conf_threshold"]),
            str(r["band_half_width"]),
        )
        for r in best_rows
    }
    for row in trial_rows:
        key = (
            str(row["model_tag"]),
            int(row["model_step"]),
            str(row["model_path"]),
            str(row["method"]),
            str(row["margin_threshold"]),
            str(row["argmax_conf_threshold"]),
            str(row["band_half_width"]),
        )
        row["is_best_for_method"] = key in best_keys

    write_csv(trial_rows, args.out_csv)
    print(f"Saved strategy search trials: {args.out_csv}")

    write_jsonl(args.out_jsonl, best_rows)
    print(f"Saved method-wise best settings: {args.out_jsonl}")


if __name__ == "__main__":
    main()
