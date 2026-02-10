from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .config import AuditConfig
from .text import basic_surface_features


def _best_threshold_accuracy(values: List[float], labels: List[int]) -> Tuple[float, float, str]:
    """Return best single-feature classification accuracy using threshold rules."""

    if not values:
        return 0.0, 0.0, "value>=thr"

    pairs = sorted(zip(values, labels), key=lambda x: x[0])
    n = len(pairs)

    prefix_pos = [0] * (n + 1)
    for i, (_, lab) in enumerate(pairs, start=1):
        prefix_pos[i] = prefix_pos[i - 1] + (1 if lab == 1 else 0)

    total_pos = prefix_pos[n]
    total_neg = n - total_pos

    best_acc = -1.0
    best_thr = float(pairs[0][0])
    best_rule = "value>=thr"

    def threshold_at_split(i: int) -> float:
        if i <= 0:
            return float(pairs[0][0]) - 1e-6
        if i >= n:
            return float(pairs[-1][0]) + 1e-6
        return float((pairs[i - 1][0] + pairs[i][0]) / 2.0)

    for i in range(0, n + 1):
        left_pos = prefix_pos[i]
        left_neg = i - left_pos
        right_pos = total_pos - left_pos
        right_neg = total_neg - left_neg

        # Predict positive when value >= threshold (right side positive)
        acc_high = (left_neg + right_pos) / n
        if acc_high > best_acc:
            best_acc = float(acc_high)
            best_thr = threshold_at_split(i)
            best_rule = "value>=thr"

        # Predict positive when value <= threshold (left side positive)
        acc_low = (left_pos + right_neg) / n
        if acc_low > best_acc:
            best_acc = float(acc_low)
            best_thr = threshold_at_split(i)
            best_rule = "value<=thr"

    return best_acc, best_thr, best_rule


def surface_signal_audit(rows: Sequence[Dict[str, Any]], cfg: AuditConfig) -> Dict[str, Any]:
    """Audit whether chosen/rejected labels are separable by shallow surface signals."""

    examples: List[Dict[str, Any]] = []
    for row in rows:
        chosen = str(row.get("chosen") or "")
        rejected = str(row.get("rejected") or "")
        if chosen:
            examples.append({"label": 1, "text": chosen})
        if rejected:
            examples.append({"label": 0, "text": rejected})

    if not examples:
        return {
            "num_pairs": len(rows),
            "num_examples": 0,
            "max_feature_accuracy": 0.0,
            "max_feature_name": None,
            "is_flagged": False,
            "threshold": cfg.separability_threshold,
            "features": [],
        }

    feature_rows = [basic_surface_features(x["text"]) for x in examples]
    feature_names = sorted(feature_rows[0].keys())
    labels = [int(x["label"]) for x in examples]

    feature_reports: List[Dict[str, Any]] = []
    best_name = None
    best_acc = -1.0

    for name in feature_names:
        vals = [float(fr[name]) for fr in feature_rows]
        acc, thr, rule = _best_threshold_accuracy(vals, labels)
        rep = {
            "feature": name,
            "best_accuracy": acc,
            "best_threshold": thr,
            "rule": rule,
        }
        feature_reports.append(rep)
        if acc > best_acc:
            best_acc = acc
            best_name = name

    feature_reports.sort(key=lambda x: x["best_accuracy"], reverse=True)

    return {
        "num_pairs": len(rows),
        "num_examples": len(examples),
        "max_feature_accuracy": float(best_acc),
        "max_feature_name": best_name,
        "is_flagged": bool(best_acc >= cfg.separability_threshold),
        "threshold": cfg.separability_threshold,
        "features": feature_reports,
    }
