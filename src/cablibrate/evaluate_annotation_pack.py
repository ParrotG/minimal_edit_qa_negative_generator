from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dataio import write_json, write_jsonl
from project_config import PROJECT_SETTINGS
from qa_judge.config import NLIConfig
from qa_judge.nli import NLIVerifier

from model_eval.correctness_transformer_matcher import (
    AnswerEquivalenceTransformerMatcher,
    TransformerMatcherConfig,
)

from .common import (
    binary_stats,
    build_values,
    group_rows_by_task_and_model,
    iter_grid,
    load_annotation_rows,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Human-annotated JSONL or DatasetDict path.")
    parser.add_argument("--split", type=str, default="train", help="Split name when data_path is a DatasetDict.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum rows to evaluate. -1 means all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit_mode", type=str, default=PROJECT_SETTINGS.calibration.fit_mode, choices=["per_model", "global"])
    parser.add_argument(
        "--search_objective",
        type=str,
        default=PROJECT_SETTINGS.calibration.search_objective,
        choices=["cohen_kappa", "f1", "accuracy"],
    )
    parser.add_argument("--reject_alpha", type=float, default=PROJECT_SETTINGS.calibration.reject_alpha)

    parser.add_argument("--nli_model_name", type=str, default=NLIConfig.model_name)
    parser.add_argument("--nli_device", type=str, default=NLIConfig.device)
    parser.add_argument("--nli_batch_size", type=int, default=NLIConfig.batch_size)
    parser.add_argument("--nli_max_length", type=int, default=NLIConfig.max_length)
    parser.add_argument("--nli_fp16", action=argparse.BooleanOptionalAction, default=NLIConfig.fp16)
    parser.add_argument("--temperature_min", type=float, default=0.05)
    parser.add_argument("--temperature_max", type=float, default=5.0)
    parser.add_argument("--temperature_step", type=float, default=0.01)

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

    parser.add_argument("--matcher_model_name", type=str, default=TransformerMatcherConfig.model_name)
    parser.add_argument("--matcher_threshold_values", type=str, default="")
    parser.add_argument("--matcher_threshold_min", type=float, default=PROJECT_SETTINGS.calibration.matcher_threshold_min)
    parser.add_argument("--matcher_threshold_max", type=float, default=PROJECT_SETTINGS.calibration.matcher_threshold_max)
    parser.add_argument("--matcher_threshold_step", type=float, default=PROJECT_SETTINGS.calibration.matcher_threshold_step)

    parser.add_argument("--out_jsonl", type=str, required=True, help="Per-row scored output JSONL.")
    parser.add_argument("--summary_csv", type=str, required=True, help="Summary CSV output.")
    parser.add_argument("--summary_json", type=str, default="", help="Optional summary JSON output.")
    parser.add_argument("--best_out", type=str, required=True, help="Best-parameter JSON output.")
    return parser.parse_args()


def _to_logit(prob: float, eps: float = 1e-12) -> float:
    q = min(max(float(prob), eps), 1.0 - eps)
    return math.log(q / (1.0 - q))


def _label_by_argmax(entail: float, neutral: float, contradict: float) -> str:
    if entail >= neutral and entail >= contradict:
        return "entail"
    if neutral >= entail and neutral >= contradict:
        return "neutral"
    return "contradict"


def _to_log_probs(entail: float, neutral: float, contradict: float) -> List[float]:
    eps = 1e-12
    return [math.log(max(entail, eps)), math.log(max(neutral, eps)), math.log(max(contradict, eps))]


def _softmax(xs: Sequence[float]) -> List[float]:
    offset = max(xs)
    exps = [math.exp(x - offset) for x in xs]
    total = sum(exps)
    return [value / total for value in exps]


def _binary_nll(rows: Sequence[Dict[str, Any]], temperature: float) -> float:
    if temperature <= 0.0:
        return float("inf")

    eps = 1e-12
    losses: List[float] = []
    for row in rows:
        label = row.get("human_label_bool")
        if label is None:
            continue
        log_probs = _to_log_probs(row["nli_entail"], row["nli_neutral"], row["nli_contradict"])
        p_entail = _softmax([value / temperature for value in log_probs])[0]
        p_entail = min(max(p_entail, eps), 1.0 - eps)
        y = 1.0 if bool(label) else 0.0
        losses.append(-(y * math.log(p_entail) + (1.0 - y) * math.log(1.0 - p_entail)))
    if not losses:
        return float("inf")
    return float(sum(losses) / len(losses))


def _fit_temperature(rows: Sequence[Dict[str, Any]], temperatures: Sequence[float]) -> Dict[str, float]:
    best_temperature = 1.0
    best_loss = float("inf")
    for value in temperatures:
        loss = _binary_nll(rows, value)
        if loss < best_loss:
            best_temperature = float(value)
            best_loss = float(loss)
    return {"temperature": best_temperature, "fit_nll": best_loss}


def _apply_temperature(row: Dict[str, Any], temperature: float) -> Dict[str, Any]:
    scaled = [value / temperature for value in _to_log_probs(row["nli_entail"], row["nli_neutral"], row["nli_contradict"])]
    entail, neutral, contradict = _softmax(scaled)
    return {
        **row,
        "temperature": float(temperature),
        "nli_entail_calib": float(entail),
        "nli_neutral_calib": float(neutral),
        "nli_contradict_calib": float(contradict),
        "nli_argmax_label_calib": _label_by_argmax(entail, neutral, contradict),
        "nli_argmax_prob_calib": float(max(entail, neutral, contradict)),
        "nli_margin_logit_entail_vs_max_other_calib": float(scaled[0] - max(scaled[1], scaled[2])),
    }


def _evaluate_predictions(
    *,
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[Optional[bool]],
    label_field: str,
    reject_alpha: float,
) -> Dict[str, float]:
    preds: List[bool] = []
    refs: List[bool] = []
    total_labeled = 0
    kept = 0
    for row, pred in zip(rows, predictions):
        label = row.get(label_field)
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
        **metrics,
    }


def _method_argmax(rows: Sequence[Dict[str, Any]]) -> List[Optional[bool]]:
    return [str(row["nli_argmax_label_calib"]) == "entail" for row in rows]


def _method_margin(rows: Sequence[Dict[str, Any]], threshold: float) -> List[Optional[bool]]:
    return [float(row["nli_margin_logit_entail_vs_max_other_calib"]) >= threshold for row in rows]


def _method_argmax_with_reject(rows: Sequence[Dict[str, Any]], min_conf: float) -> List[Optional[bool]]:
    out: List[Optional[bool]] = []
    base = _method_argmax(rows)
    for row, pred in zip(rows, base):
        out.append(pred if float(row["nli_argmax_prob_calib"]) >= min_conf else None)
    return out


def _method_margin_with_reject_band(rows: Sequence[Dict[str, Any]], threshold: float, half_width: float) -> List[Optional[bool]]:
    out: List[Optional[bool]] = []
    for row in rows:
        margin = float(row["nli_margin_logit_entail_vs_max_other_calib"])
        if abs(margin - threshold) <= half_width:
            out.append(None)
        else:
            out.append(margin >= threshold)
    return out


def _search_nli_group(
    *,
    rows: Sequence[Dict[str, Any]],
    objective: str,
    reject_alpha: float,
    margin_thresholds: Sequence[float],
    argmax_conf_thresholds: Sequence[float],
    band_half_widths: Sequence[float],
    group_meta: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    search_rows: List[Dict[str, Any]] = []

    argmax_predictions = _method_argmax(rows)
    search_rows.append(
        {
            **group_meta,
            "method": "argmax",
            **_evaluate_predictions(rows=rows, predictions=argmax_predictions, label_field="human_label_bool", reject_alpha=reject_alpha),
        }
    )
    for threshold in margin_thresholds:
        search_rows.append(
            {
                **group_meta,
                "method": "margin",
                "margin_threshold": float(threshold),
                **_evaluate_predictions(
                    rows=rows,
                    predictions=_method_margin(rows, threshold=float(threshold)),
                    label_field="human_label_bool",
                    reject_alpha=reject_alpha,
                ),
            }
        )
    for threshold in argmax_conf_thresholds:
        search_rows.append(
            {
                **group_meta,
                "method": "argmax_with_reject",
                "argmax_conf_threshold": float(threshold),
                **_evaluate_predictions(
                    rows=rows,
                    predictions=_method_argmax_with_reject(rows, min_conf=float(threshold)),
                    label_field="human_label_bool",
                    reject_alpha=reject_alpha,
                ),
            }
        )
    for margin_threshold, half_width in iter_grid(margin_thresholds, band_half_widths):
        search_rows.append(
            {
                **group_meta,
                "method": "margin_with_reject_band",
                "margin_threshold": float(margin_threshold),
                "band_half_width": float(half_width),
                **_evaluate_predictions(
                    rows=rows,
                    predictions=_method_margin_with_reject_band(rows, threshold=float(margin_threshold), half_width=float(half_width)),
                    label_field="human_label_bool",
                    reject_alpha=reject_alpha,
                ),
            }
        )

    minimize = False
    best_row = max(search_rows, key=lambda row: (float(row.get(objective, float("-inf"))), float(row.get("coverage", float("-inf")))))
    reject_candidates = [row for row in search_rows if "reject" in str(row.get("method") or "")]
    if reject_candidates:
        best_reject = min(
            reject_candidates,
            key=lambda row: (float(row.get("lagrangian_objective", float("inf"))), -float(row.get("coverage", float("-inf")))),
        )
        best_row = best_reject if math.isfinite(float(best_reject.get("lagrangian_objective", float("inf")))) else best_row
        minimize = True
    return search_rows, {**best_row, "best_sort_minimize": minimize}


def _search_matcher_group(
    *,
    rows: Sequence[Dict[str, Any]],
    thresholds: Sequence[float],
    objective: str,
    group_meta: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    search_rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        predictions = [float(row.get("matcher_match_score") or 0.0) >= float(threshold) for row in rows]
        metrics = _evaluate_predictions(rows=rows, predictions=predictions, label_field="human_label_bool", reject_alpha=0.0)
        search_rows.append(
            {
                **group_meta,
                "method": "matcher_threshold",
                "matcher_threshold": float(threshold),
                **metrics,
            }
        )
    best_row = max(search_rows, key=lambda row: (float(row.get(objective, float("-inf"))), float(row.get("accuracy", float("-inf")))))
    return search_rows, best_row


def _score_nli_rows(rows: Sequence[Dict[str, Any]], verifier: NLIVerifier) -> List[Dict[str, Any]]:
    if not rows:
        return []
    scores = verifier.score(
        [str(row.get("premise_text") or "") for row in rows],
        [str(row.get("hypothesis_text") or "") for row in rows],
    )
    out_rows: List[Dict[str, Any]] = []
    for row, score in zip(rows, scores):
        entail = float(score.entail)
        neutral = float(score.neutral)
        contradict = float(score.contradict)
        out_rows.append(
            {
                **row,
                "nli_entail": entail,
                "nli_neutral": neutral,
                "nli_contradict": contradict,
                "nli_argmax_label": _label_by_argmax(entail, neutral, contradict),
                "nli_argmax_prob": float(max(entail, neutral, contradict)),
                "nli_logit_entail": _to_logit(entail),
                "nli_logit_neutral": _to_logit(neutral),
                "nli_logit_contradict": _to_logit(contradict),
                "nli_margin_logit_entail_vs_max_other": float(_to_logit(entail) - max(_to_logit(neutral), _to_logit(contradict))),
            }
        )
    return out_rows


def _score_matcher_rows(rows: Sequence[Dict[str, Any]], matcher: AnswerEquivalenceTransformerMatcher) -> List[Dict[str, Any]]:
    if not rows:
        return []
    reports = matcher.review_batch(rows)
    return [
        {
            **row,
            "matcher_ok": report.ok,
            "matcher_match_score": report.match_score,
        }
        for row, report in zip(rows, reports)
    ]


def _fit_nli_temperatures(
    *,
    rows: Sequence[Dict[str, Any]],
    fit_mode: str,
    temperature_values: Sequence[float],
) -> Dict[Tuple[str, str, int, str], Dict[str, float]]:
    grouped = group_rows_by_task_and_model(rows)
    fit_map: Dict[Tuple[str, str, int, str], Dict[str, float]] = {}
    if fit_mode == "global":
        task_groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            task_groups.setdefault(str(row.get("task_type") or ""), []).append(row)
        global_fit = {
            task_type: _fit_temperature(
                [row for row in task_rows if row.get("human_label_bool") is not None],
                temperature_values,
            )
            for task_type, task_rows in task_groups.items()
        }
        for key in grouped:
            fit_map[key] = dict(global_fit.get(key[0]) or {"temperature": 1.0, "fit_nll": float("inf")})
        return fit_map

    for key, group_rows in grouped.items():
        labeled_rows = [row for row in group_rows if row.get("human_label_bool") is not None]
        fit_map[key] = _fit_temperature(labeled_rows, temperature_values)
    return fit_map


def main() -> None:
    args = parse_args()
    rows = load_annotation_rows(
        data_path=args.data_path,
        split=args.split,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    if not rows:
        raise RuntimeError("No valid annotation rows found.")

    nli_rows = [row for row in rows if str(row.get("task_type") or "") in {"nli_flat", "nli_structured"}]
    matcher_rows = [row for row in rows if str(row.get("task_type") or "") == "matcher"]

    scored_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []

    if nli_rows:
        verifier = NLIVerifier(
            model_name=args.nli_model_name,
            device=args.nli_device,
            batch_size=args.nli_batch_size,
            max_length=args.nli_max_length,
            fp16=args.nli_fp16,
        )
        scored_nli_rows = _score_nli_rows(nli_rows, verifier)
        temperature_values = build_values(
            explicit="",
            v_min=args.temperature_min,
            v_max=args.temperature_max,
            v_step=args.temperature_step,
            name="temperature",
        )
        fit_map = _fit_nli_temperatures(rows=scored_nli_rows, fit_mode=args.fit_mode, temperature_values=temperature_values)
        grouped_nli_rows = group_rows_by_task_and_model(scored_nli_rows)
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
        for key, group_rows in grouped_nli_rows.items():
            fit_payload = fit_map[key]
            calibrated_rows = [_apply_temperature(row, float(fit_payload["temperature"])) for row in group_rows]
            scored_rows.extend(calibrated_rows)
            group_meta = {
                "task_type": key[0],
                "model_tag": key[1],
                "model_step": key[2],
                "model_path": key[3],
                "temperature": float(fit_payload["temperature"]),
                "fit_nll": float(fit_payload["fit_nll"]),
            }
            group_search_rows, best_row = _search_nli_group(
                rows=calibrated_rows,
                objective=args.search_objective,
                reject_alpha=args.reject_alpha,
                margin_thresholds=margin_thresholds,
                argmax_conf_thresholds=argmax_conf_thresholds,
                band_half_widths=band_half_widths,
                group_meta=group_meta,
            )
            summary_rows.extend(group_search_rows)
            best_rows.append(best_row)

    if matcher_rows:
        matcher = AnswerEquivalenceTransformerMatcher(
            TransformerMatcherConfig(model_name=args.matcher_model_name)
        )
        scored_matcher_rows = _score_matcher_rows(matcher_rows, matcher)
        scored_rows.extend(scored_matcher_rows)
        grouped_matcher_rows = group_rows_by_task_and_model(scored_matcher_rows)
        matcher_thresholds = build_values(
            explicit=args.matcher_threshold_values,
            v_min=args.matcher_threshold_min,
            v_max=args.matcher_threshold_max,
            v_step=args.matcher_threshold_step,
            name="matcher_threshold",
        )
        for key, group_rows in grouped_matcher_rows.items():
            group_meta = {
                "task_type": key[0],
                "model_tag": key[1],
                "model_step": key[2],
                "model_path": key[3],
            }
            group_search_rows, best_row = _search_matcher_group(
                rows=group_rows,
                thresholds=matcher_thresholds,
                objective=args.search_objective,
                group_meta=group_meta,
            )
            summary_rows.extend(group_search_rows)
            best_rows.append(best_row)

    write_jsonl(args.out_jsonl, scored_rows)
    write_csv(summary_rows, args.summary_csv)
    write_json(args.best_out, {"rows": best_rows})
    if str(args.summary_json or "").strip():
        write_json(
            args.summary_json,
            {
                "num_rows": len(rows),
                "num_scored_rows": len(scored_rows),
                "num_summary_rows": len(summary_rows),
                "num_best_rows": len(best_rows),
                "summary_rows": summary_rows,
                "best_rows": best_rows,
            },
        )
    print(f"Saved per-row calibration evaluation: {args.out_jsonl}")


if __name__ == "__main__":
    main()
