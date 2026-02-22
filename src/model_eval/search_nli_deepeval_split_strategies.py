from __future__ import annotations

import argparse
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dataio import write_jsonl

from .common import group_rows_by_model, load_dataset_split, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nli_scored_path",
        type=str,
        required=True,
        help="Path to NLI-scored rows from score_nli_from_deepeval_details.py.",
    )
    parser.add_argument("--split", type=str, default="train", help="Split name when input path is a DatasetDict.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum rows to evaluate. -1 means all.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--search_objective",
        type=str,
        default="cohen_kappa",
        choices=["cohen_kappa", "faithful_f1", "nli_accuracy_vs_deepeval"],
        help="Objective to maximize for non-reject strategies.",
    )
    parser.add_argument(
        "--reject_alpha",
        type=float,
        default=0.25,
        help="Lagrangian alpha in Error(kept) + alpha * RejectRate for reject strategies.",
    )

    parser.add_argument(
        "--margin_threshold_values",
        type=str,
        default="",
        help="Comma-separated thresholds for logit-margin strategy. Overrides min/max/step when set.",
    )
    parser.add_argument("--margin_threshold_min", type=float, default=-4.0)
    parser.add_argument("--margin_threshold_max", type=float, default=4.0)
    parser.add_argument("--margin_threshold_step", type=float, default=0.1)

    parser.add_argument(
        "--argmax_conf_threshold_values",
        type=str,
        default="",
        help="Comma-separated confidence thresholds for argmax+reject strategy. Overrides min/max/step when set.",
    )
    parser.add_argument("--argmax_conf_threshold_min", type=float, default=0.34)
    parser.add_argument("--argmax_conf_threshold_max", type=float, default=0.99)
    parser.add_argument("--argmax_conf_threshold_step", type=float, default=0.01)

    parser.add_argument(
        "--band_half_width_values",
        type=str,
        default="",
        help="Comma-separated half-width values for margin-band reject strategy.",
    )
    parser.add_argument("--band_half_width_min", type=float, default=0.0)
    parser.add_argument("--band_half_width_max", type=float, default=1.0)
    parser.add_argument("--band_half_width_step", type=float, default=0.05)

    parser.add_argument("--out_csv", type=str, default="nli_deepeval_split_search.csv")
    parser.add_argument("--out_jsonl", type=str, default="nli_deepeval_split_search_best.jsonl")
    return parser.parse_args()


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _binary_stats(preds: Sequence[bool], refs: Sequence[bool]) -> Dict[str, float]:
    n = len(preds)
    if n == 0:
        return {
            "accuracy": float("nan"),
            "cohen_kappa": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "tp": 0.0,
            "tn": 0.0,
            "fp": 0.0,
            "fn": 0.0,
        }

    tp = sum(1 for p, r in zip(preds, refs) if p and r)
    tn = sum(1 for p, r in zip(preds, refs) if (not p) and (not r))
    fp = sum(1 for p, r in zip(preds, refs) if p and (not r))
    fn = sum(1 for p, r in zip(preds, refs) if (not p) and r)

    accuracy = (tp + tn) / n
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)

    p_yes_pred = sum(1 for x in preds if x) / n
    p_yes_ref = sum(1 for x in refs if x) / n
    p_e = p_yes_pred * p_yes_ref + (1.0 - p_yes_pred) * (1.0 - p_yes_ref)
    kappa = float("nan") if abs(1.0 - p_e) < 1e-12 else (accuracy - p_e) / (1.0 - p_e)

    return {
        "accuracy": float(accuracy),
        "cohen_kappa": float(kappa),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


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


def _load_scored_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    ds = load_dataset_split(data_path=args.nli_scored_path, split=args.split).shuffle(seed=args.seed)

    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        entail = _safe_float(row.get("nli_entail"))
        neutral = _safe_float(row.get("nli_neutral"))
        contradict = _safe_float(row.get("nli_contradict"))
        if entail is None or neutral is None or contradict is None:
            continue

        argmax_prob = _safe_float(row.get("nli_argmax_prob"))
        if argmax_prob is None:
            argmax_prob = max(entail, neutral, contradict)

        margin = _safe_float(row.get("nli_margin_logit_entail_vs_max_other"))
        if margin is None:
            eps = 1e-12
            l_ent = math.log(min(max(entail, eps), 1.0 - eps) / (1.0 - min(max(entail, eps), 1.0 - eps)))
            l_neu = math.log(min(max(neutral, eps), 1.0 - eps) / (1.0 - min(max(neutral, eps), 1.0 - eps)))
            l_con = math.log(min(max(contradict, eps), 1.0 - eps) / (1.0 - min(max(contradict, eps), 1.0 - eps)))
            margin = l_ent - max(l_neu, l_con)

        deepeval_success = _safe_bool(row.get("deepeval_success"))
        out.append(
            {
                "model_tag": str(row.get("model_tag") or "model"),
                "model_step": int(row.get("model_step", 0)),
                "model_path": str(row.get("model_path") or row.get("model_tag") or "model"),
                "sample_id": int(row.get("sample_id", idx)),
                "source_id": str(row.get("source_id") or row.get("sample_id") or idx),
                "deepeval_success": deepeval_success,
                "nli_entail": entail,
                "nli_neutral": neutral,
                "nli_contradict": contradict,
                "nli_argmax_label": str(row.get("nli_argmax_label") or ""),
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
    comparable_preds: List[bool] = []
    comparable_refs: List[bool] = []

    total_comparable = 0
    kept = 0
    for row, pred in zip(rows, predictions):
        ref = row.get("deepeval_success")
        if ref is None:
            continue
        total_comparable += 1
        if pred is None:
            continue
        kept += 1
        comparable_preds.append(bool(pred))
        comparable_refs.append(bool(ref))

    reject_count = total_comparable - kept
    reject_rate = float(reject_count / total_comparable) if total_comparable > 0 else float("nan")
    coverage = float(kept / total_comparable) if total_comparable > 0 else float("nan")

    bstats = _binary_stats(comparable_preds, comparable_refs)
    error_kept = 1.0 - bstats["accuracy"] if kept > 0 else 1.0
    lagrangian = error_kept + reject_alpha * reject_rate if total_comparable > 0 else float("nan")

    return {
        "num_rows": float(len(rows)),
        "num_comparable": float(total_comparable),
        "num_kept": float(kept),
        "num_rejected": float(reject_count),
        "coverage": coverage,
        "reject_rate": reject_rate,
        "error_kept": float(error_kept),
        "lagrangian_objective": float(lagrangian),
        "nli_accuracy_vs_deepeval": bstats["accuracy"],
        "cohen_kappa": bstats["cohen_kappa"],
        "faithful_precision": bstats["precision"],
        "faithful_recall": bstats["recall"],
        "faithful_f1": bstats["f1"],
        "tp": bstats["tp"],
        "tn": bstats["tn"],
        "fp": bstats["fp"],
        "fn": bstats["fn"],
    }


def _method_argmax(rows: Sequence[Dict[str, Any]]) -> List[Optional[bool]]:
    out: List[Optional[bool]] = []
    for row in rows:
        label = row["nli_argmax_label"]
        if not label:
            entail = float(row["nli_entail"])
            neutral = float(row["nli_neutral"])
            contradict = float(row["nli_contradict"])
            if entail >= neutral and entail >= contradict:
                label = "entail"
            elif neutral >= entail and neutral >= contradict:
                label = "neutral"
            else:
                label = "contradict"
        out.append(label == "entail")
    return out


def _method_margin(rows: Sequence[Dict[str, Any]], threshold: float) -> List[Optional[bool]]:
    return [float(row["nli_margin_logit_entail_vs_max_other"]) >= threshold for row in rows]


def _method_argmax_with_reject(rows: Sequence[Dict[str, Any]], min_conf: float) -> List[Optional[bool]]:
    out: List[Optional[bool]] = []
    base = _method_argmax(rows)
    for row, pred in zip(rows, base):
        conf = float(row["nli_argmax_prob"])
        out.append(pred if conf >= min_conf else None)
    return out


def _method_margin_with_band_reject(rows: Sequence[Dict[str, Any]], threshold: float, half_width: float) -> List[Optional[bool]]:
    out: List[Optional[bool]] = []
    for row in rows:
        score = float(row["nli_margin_logit_entail_vs_max_other"])
        if abs(score - threshold) <= half_width:
            out.append(None)
        else:
            out.append(score >= threshold)
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


def _iter_grid(a_values: Iterable[float], b_values: Iterable[float]) -> Iterable[Tuple[float, float]]:
    for a in a_values:
        for b in b_values:
            yield float(a), float(b)


def main() -> None:
    args = parse_args()
    rows = _load_scored_rows(args)
    if not rows:
        raise RuntimeError("No valid rows found from NLI-scored input.")

    margin_thresholds = _build_values(
        explicit=args.margin_threshold_values,
        v_min=args.margin_threshold_min,
        v_max=args.margin_threshold_max,
        v_step=args.margin_threshold_step,
        name="margin_threshold",
    )
    argmax_conf_thresholds = _build_values(
        explicit=args.argmax_conf_threshold_values,
        v_min=args.argmax_conf_threshold_min,
        v_max=args.argmax_conf_threshold_max,
        v_step=args.argmax_conf_threshold_step,
        name="argmax_conf_threshold",
    )
    band_half_widths = _build_values(
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

        method_buckets: Dict[str, List[Dict[str, Any]]] = {
            "argmax_entail": [],
            "logit_margin_threshold": [],
            "argmax_with_conf_reject": [],
            "margin_with_band_reject": [],
        }

        preds_argmax = _method_argmax(model_rows)
        metrics_argmax = _evaluate_predictions(rows=model_rows, predictions=preds_argmax, reject_alpha=args.reject_alpha)
        row_argmax = {
            "model_tag": model_tag,
            "model_step": model_step,
            "model_path": model_path,
            "method": "argmax_entail",
            "margin_threshold": "",
            "argmax_conf_threshold": "",
            "band_half_width": "",
            "search_objective": args.search_objective,
            "search_objective_value": metrics_argmax[args.search_objective],
            "reject_alpha": args.reject_alpha,
            **metrics_argmax,
        }
        method_buckets["argmax_entail"].append(row_argmax)
        trial_rows.append(row_argmax)

        for th in margin_thresholds:
            preds = _method_margin(model_rows, threshold=th)
            metrics = _evaluate_predictions(rows=model_rows, predictions=preds, reject_alpha=args.reject_alpha)
            row = {
                "model_tag": model_tag,
                "model_step": model_step,
                "model_path": model_path,
                "method": "logit_margin_threshold",
                "margin_threshold": th,
                "argmax_conf_threshold": "",
                "band_half_width": "",
                "search_objective": args.search_objective,
                "search_objective_value": metrics[args.search_objective],
                "reject_alpha": args.reject_alpha,
                **metrics,
            }
            method_buckets["logit_margin_threshold"].append(row)
            trial_rows.append(row)

        for conf in argmax_conf_thresholds:
            preds = _method_argmax_with_reject(model_rows, min_conf=conf)
            metrics = _evaluate_predictions(rows=model_rows, predictions=preds, reject_alpha=args.reject_alpha)
            row = {
                "model_tag": model_tag,
                "model_step": model_step,
                "model_path": model_path,
                "method": "argmax_with_conf_reject",
                "margin_threshold": "",
                "argmax_conf_threshold": conf,
                "band_half_width": "",
                "search_objective": args.search_objective,
                "search_objective_value": metrics[args.search_objective],
                "reject_alpha": args.reject_alpha,
                **metrics,
            }
            method_buckets["argmax_with_conf_reject"].append(row)
            trial_rows.append(row)

        for th, half_w in _iter_grid(margin_thresholds, band_half_widths):
            preds = _method_margin_with_band_reject(model_rows, threshold=th, half_width=half_w)
            metrics = _evaluate_predictions(rows=model_rows, predictions=preds, reject_alpha=args.reject_alpha)
            row = {
                "model_tag": model_tag,
                "model_step": model_step,
                "model_path": model_path,
                "method": "margin_with_band_reject",
                "margin_threshold": th,
                "argmax_conf_threshold": "",
                "band_half_width": half_w,
                "search_objective": args.search_objective,
                "search_objective_value": metrics[args.search_objective],
                "reject_alpha": args.reject_alpha,
                **metrics,
            }
            method_buckets["margin_with_band_reject"].append(row)
            trial_rows.append(row)

        for method_name, candidates in method_buckets.items():
            if method_name in {"argmax_with_conf_reject", "margin_with_band_reject"}:
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
    print(f"Saved search trials: {args.out_csv}")

    write_jsonl(args.out_jsonl, best_rows)
    print(f"Saved method-wise best settings: {args.out_jsonl}")


if __name__ == "__main__":
    main()

