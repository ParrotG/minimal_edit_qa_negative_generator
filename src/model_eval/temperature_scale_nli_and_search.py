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
        help="Input path from score_nli_from_deepeval_details.py.",
    )
    parser.add_argument("--split", type=str, default="train", help="Split name when input path is a DatasetDict.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum rows to process. -1 means all.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--fit_mode",
        type=str,
        default="per_model",
        choices=["per_model", "global"],
        help="Temperature fitting scope.",
    )
    parser.add_argument("--temperature_min", type=float, default=0.05)
    parser.add_argument("--temperature_max", type=float, default=5.0)
    parser.add_argument("--temperature_step", type=float, default=0.01)

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

    parser.add_argument("--margin_threshold_values", type=str, default="")
    parser.add_argument("--margin_threshold_min", type=float, default=-4.0)
    parser.add_argument("--margin_threshold_max", type=float, default=4.0)
    parser.add_argument("--margin_threshold_step", type=float, default=0.1)

    parser.add_argument("--argmax_conf_threshold_values", type=str, default="")
    parser.add_argument("--argmax_conf_threshold_min", type=float, default=0.34)
    parser.add_argument("--argmax_conf_threshold_max", type=float, default=0.99)
    parser.add_argument("--argmax_conf_threshold_step", type=float, default=0.01)

    parser.add_argument("--band_half_width_values", type=str, default="")
    parser.add_argument("--band_half_width_min", type=float, default=0.0)
    parser.add_argument("--band_half_width_max", type=float, default=1.0)
    parser.add_argument("--band_half_width_step", type=float, default=0.05)

    parser.add_argument("--out_calibrated_jsonl", type=str, default="nli_scored_calibrated.jsonl")
    parser.add_argument("--out_temperature_csv", type=str, default="nli_temperature_scaling.csv")
    parser.add_argument("--out_search_csv", type=str, default="nli_deepeval_split_search_calibrated.csv")
    parser.add_argument("--out_search_best_jsonl", type=str, default="nli_deepeval_split_search_calibrated_best.jsonl")
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


def _iter_grid(a_values: Iterable[float], b_values: Iterable[float]) -> Iterable[Tuple[float, float]]:
    for a in a_values:
        for b in b_values:
            yield float(a), float(b)


def _softmax(xs: Sequence[float]) -> List[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]


def _to_logits_from_probs(entail: float, neutral: float, contradict: float) -> Tuple[float, float, float]:
    eps = 1e-12
    return (
        math.log(max(entail, eps)),
        math.log(max(neutral, eps)),
        math.log(max(contradict, eps)),
    )


def _load_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    ds = load_dataset_split(data_path=args.nli_scored_path, split=args.split).shuffle(seed=args.seed)

    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        entail = _safe_float(row.get("nli_entail"))
        neutral = _safe_float(row.get("nli_neutral"))
        contradict = _safe_float(row.get("nli_contradict"))
        if entail is None or neutral is None or contradict is None:
            continue

        out.append(
            {
                "model_tag": str(row.get("model_tag") or "model"),
                "model_step": int(row.get("model_step", 0)),
                "model_path": str(row.get("model_path") or row.get("model_tag") or "model"),
                "sample_id": int(row.get("sample_id", idx)),
                "source_id": str(row.get("source_id") or row.get("sample_id") or idx),
                "question": str(row.get("question") or ""),
                "knowledge": str(row.get("knowledge") or ""),
                "answer": str(row.get("answer") or ""),
                "deepeval_success": _safe_bool(row.get("deepeval_success")),
                "deepeval_score": _safe_float(row.get("deepeval_score")),
                "nli_entail": float(entail),
                "nli_neutral": float(neutral),
                "nli_contradict": float(contradict),
            }
        )
        if args.max_samples > 0 and len(out) >= args.max_samples:
            break
    return out


def _binary_nll_for_temperature(rows: Sequence[Dict[str, Any]], temperature: float) -> float:
    if temperature <= 0.0:
        return float("inf")

    losses: List[float] = []
    eps = 1e-12
    for row in rows:
        y = row.get("deepeval_success")
        if y is None:
            continue
        l_ent, l_neu, l_con = _to_logits_from_probs(row["nli_entail"], row["nli_neutral"], row["nli_contradict"])
        p_ent = _softmax([l_ent / temperature, l_neu / temperature, l_con / temperature])[0]
        p_ent = min(max(p_ent, eps), 1.0 - eps)
        target = 1.0 if bool(y) else 0.0
        losses.append(-(target * math.log(p_ent) + (1.0 - target) * math.log(1.0 - p_ent)))

    if not losses:
        return float("inf")
    return float(sum(losses) / len(losses))


def _fit_temperature(rows: Sequence[Dict[str, Any]], t_values: Sequence[float]) -> Tuple[float, float]:
    best_t = 1.0
    best_loss = float("inf")
    for t in t_values:
        loss = _binary_nll_for_temperature(rows, t)
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    return best_t, best_loss


def _apply_temperature(row: Dict[str, Any], temperature: float) -> Dict[str, Any]:
    l_ent, l_neu, l_con = _to_logits_from_probs(row["nli_entail"], row["nli_neutral"], row["nli_contradict"])
    p_ent, p_neu, p_con = _softmax([l_ent / temperature, l_neu / temperature, l_con / temperature])

    if p_ent >= p_neu and p_ent >= p_con:
        label = "entail"
    elif p_neu >= p_ent and p_neu >= p_con:
        label = "neutral"
    else:
        label = "contradict"

    margin = (l_ent / temperature) - max(l_neu / temperature, l_con / temperature)
    return {
        **row,
        "temperature": float(temperature),
        "nli_entail_calib": float(p_ent),
        "nli_neutral_calib": float(p_neu),
        "nli_contradict_calib": float(p_con),
        "nli_argmax_label_calib": label,
        "nli_argmax_prob_calib": float(max(p_ent, p_neu, p_con)),
        "nli_margin_logit_entail_vs_max_other_calib": float(margin),
    }


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
    return [str(row["nli_argmax_label_calib"]) == "entail" for row in rows]


def _method_margin(rows: Sequence[Dict[str, Any]], threshold: float) -> List[Optional[bool]]:
    return [float(row["nli_margin_logit_entail_vs_max_other_calib"]) >= threshold for row in rows]


def _method_argmax_with_reject(rows: Sequence[Dict[str, Any]], min_conf: float) -> List[Optional[bool]]:
    out: List[Optional[bool]] = []
    base = _method_argmax(rows)
    for row, pred in zip(rows, base):
        conf = float(row["nli_argmax_prob_calib"])
        out.append(pred if conf >= min_conf else None)
    return out


def _method_margin_with_band_reject(rows: Sequence[Dict[str, Any]], threshold: float, half_width: float) -> List[Optional[bool]]:
    out: List[Optional[bool]] = []
    for row in rows:
        score = float(row["nli_margin_logit_entail_vs_max_other_calib"])
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


def main() -> None:
    args = parse_args()
    rows = _load_rows(args)
    if not rows:
        raise RuntimeError("No valid rows found from input nli_scored_path.")

    temperature_values = _build_values(
        explicit="",
        v_min=args.temperature_min,
        v_max=args.temperature_max,
        v_step=args.temperature_step,
        name="temperature",
    )
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
    temperature_rows: List[Dict[str, Any]] = []
    calibrated_rows: List[Dict[str, Any]] = []
    trial_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []

    global_temperature = None
    if args.fit_mode == "global":
        global_temperature, global_loss = _fit_temperature(rows, temperature_values)
        temperature_rows.append(
            {
                "fit_mode": "global",
                "model_tag": "__all__",
                "model_step": -1,
                "model_path": "__all__",
                "temperature": global_temperature,
                "fit_nll": global_loss,
                "num_rows": len(rows),
                "num_labeled": sum(1 for r in rows if r.get("deepeval_success") is not None),
            }
        )

    for _, model_rows in sorted(grouped.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        model_tag = str(model_rows[0]["model_tag"])
        model_step = int(model_rows[0]["model_step"])
        model_path = str(model_rows[0]["model_path"])

        if global_temperature is None:
            temp, fit_loss = _fit_temperature(model_rows, temperature_values)
        else:
            temp = float(global_temperature)
            fit_loss = _binary_nll_for_temperature(model_rows, temp)

        temperature_rows.append(
            {
                "fit_mode": args.fit_mode,
                "model_tag": model_tag,
                "model_step": model_step,
                "model_path": model_path,
                "temperature": temp,
                "fit_nll": fit_loss,
                "num_rows": len(model_rows),
                "num_labeled": sum(1 for r in model_rows if r.get("deepeval_success") is not None),
            }
        )

        model_calibrated = [_apply_temperature(row, temp) for row in model_rows]
        calibrated_rows.extend(model_calibrated)

        method_buckets: Dict[str, List[Dict[str, Any]]] = {
            "argmax_entail": [],
            "logit_margin_threshold": [],
            "argmax_with_conf_reject": [],
            "margin_with_band_reject": [],
        }

        preds_argmax = _method_argmax(model_calibrated)
        metrics_argmax = _evaluate_predictions(
            rows=model_calibrated,
            predictions=preds_argmax,
            reject_alpha=args.reject_alpha,
        )
        row_argmax = {
            "fit_mode": args.fit_mode,
            "temperature": temp,
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
            preds = _method_margin(model_calibrated, threshold=th)
            metrics = _evaluate_predictions(
                rows=model_calibrated,
                predictions=preds,
                reject_alpha=args.reject_alpha,
            )
            row = {
                "fit_mode": args.fit_mode,
                "temperature": temp,
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
            preds = _method_argmax_with_reject(model_calibrated, min_conf=conf)
            metrics = _evaluate_predictions(
                rows=model_calibrated,
                predictions=preds,
                reject_alpha=args.reject_alpha,
            )
            row = {
                "fit_mode": args.fit_mode,
                "temperature": temp,
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
            preds = _method_margin_with_band_reject(model_calibrated, threshold=th, half_width=half_w)
            metrics = _evaluate_predictions(
                rows=model_calibrated,
                predictions=preds,
                reject_alpha=args.reject_alpha,
            )
            row = {
                "fit_mode": args.fit_mode,
                "temperature": temp,
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
                    "used_temperature_range": f"{args.temperature_min}:{args.temperature_max}:{args.temperature_step}",
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

    write_jsonl(args.out_calibrated_jsonl, calibrated_rows)
    print(f"Saved calibrated rows: {args.out_calibrated_jsonl}")

    write_csv(temperature_rows, args.out_temperature_csv)
    print(f"Saved temperature summary: {args.out_temperature_csv}")

    write_csv(trial_rows, args.out_search_csv)
    print(f"Saved search trials: {args.out_search_csv}")

    write_jsonl(args.out_search_best_jsonl, best_rows)
    print(f"Saved method-wise best settings: {args.out_search_best_jsonl}")


if __name__ == "__main__":
    main()

