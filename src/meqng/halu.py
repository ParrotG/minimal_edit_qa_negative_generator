from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm

from .config import HaluEvalConfig
from .nli import NLIVerifier
from .ner import NERTagger, build_entity_bank
from .io import write_json
from .prompt import build_qa_premise


@dataclass(frozen=True)
class QASample:
    """Unified QA sample for downstream stages."""
    uid: str
    knowledge: str
    question: str
    right_answer: str
    hallucinated_answer: str


def iter_halueval_qa(cfg: HaluEvalConfig, max_samples: Optional[int] = None, seed: int = 0) -> Iterator[QASample]:
    """Iterate over HaluEval QA examples."""
    ds = load_dataset(cfg.dataset_name, cfg.subset, split=cfg.split)
    n = len(ds)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    if max_samples is not None:
        indices = indices[: max_samples]

    for i in indices:
        row = ds[i]
        yield QASample(
            uid=str(i),
            knowledge=row[cfg.knowledge_field],
            question=row[cfg.question_field],
            right_answer=row[cfg.right_answer_field],
            hallucinated_answer=row[cfg.hallucinated_answer_field],
        )


def sanity_check_nli(
    samples: Iterable[QASample],
    nli: NLIVerifier,
    out_path: str,
    entail_threshold_pos: float = 0.60,
    entail_threshold_neg: float = 0.30,
) -> Dict[str, float]:
    """Compute NLI agreement rates on right vs hallucinated answers."""
    premises: List[str] = []
    pos_h: List[str] = []
    neg_h: List[str] = []
    for s in samples:
        premise = build_qa_premise(s.knowledge, s.question)
        premises.append(premise)
        pos_h.append(s.right_answer)
        neg_h.append(s.hallucinated_answer)

    pos_scores = nli.score(premises, pos_h)
    neg_scores = nli.score(premises, neg_h)

    pos_entail = [sc.entail for sc in pos_scores]
    neg_entail = [sc.entail for sc in neg_scores]

    pos_ok = sum(1 for p in pos_entail if p >= entail_threshold_pos)
    neg_ok = sum(1 for p in neg_entail if p <= entail_threshold_neg)

    metrics = {
        "num_samples": float(len(premises)),
        "pos_entail_rate@thr": pos_ok / max(1, len(premises)),
        "neg_non_entail_rate@thr": neg_ok / max(1, len(premises)),
        "avg_pos_entail": float(sum(pos_entail) / max(1, len(pos_entail))),
        "avg_neg_entail": float(sum(neg_entail) / max(1, len(neg_entail))),
    }
    write_json(out_path, metrics)
    return metrics


def build_entity_bank_from_halueval(
    samples: Iterable[QASample],
    out_path: str,
    spacy_model: str = "en_core_web_trf",
    min_count: int = 2,
) -> Dict[str, List[Dict[str, int | str]]]:
    """Build and save an entity bank from knowledge/question/answers."""
    ner = NERTagger(spacy_model)
    texts = []
    for s in tqdm(list(samples), desc="Collecting texts"):
        texts.append(s.knowledge)
        texts.append(s.question)
        texts.append(s.right_answer)
        texts.append(s.hallucinated_answer)

    bank = build_entity_bank(texts, ner=ner, min_count=min_count)
    write_json(out_path, bank)
    return bank
