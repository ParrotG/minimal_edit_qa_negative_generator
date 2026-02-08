from __future__ import annotations

from typing import Any, Dict, Tuple


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_rank_score(filter_meta: Dict[str, Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
    """
    Compute a heuristic ranking score for kept candidates.

    Higher score means a harder-but-still-valid negative:
    - near entailment boundary (but still non-entail)
    - minimal edit distance
    - similar length
    - semantically still aligned with the original Q-A intent
    """

    components: Dict[str, float] = {}
    weights: Dict[str, float] = {}

    nli_meta = filter_meta.get("nli_flip")
    if nli_meta is not None and "cand_entail" in nli_meta:
        components["nli_hardness"] = _clip01(float(nli_meta["cand_entail"]))
        weights["nli_hardness"] = 0.45

    edit_meta = filter_meta.get("edit_distance")
    if edit_meta is not None and "norm" in edit_meta:
        components["edit_proximity"] = _clip01(1.0 - float(edit_meta["norm"]))
        weights["edit_proximity"] = 0.25

    len_meta = filter_meta.get("length_ratio")
    if len_meta is not None and "ratio" in len_meta:
        ratio = float(len_meta["ratio"])
        components["length_proximity"] = _clip01(1.0 - abs(1.0 - ratio))
        weights["length_proximity"] = 0.15

    qa_meta = filter_meta.get("qa_consistency")
    if qa_meta is not None and "similarity" in qa_meta:
        components["qa_similarity"] = _clip01(float(qa_meta["similarity"]))
        weights["qa_similarity"] = 0.15

    if not components:
        return 0.0, {}

    total_weight = sum(weights[k] for k in components.keys())
    if total_weight <= 0:
        return 0.0, components

    score = sum(components[k] * weights[k] for k in components.keys()) / total_weight
    return float(score), components
