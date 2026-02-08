from .base import PerturbCandidate, Perturbator
from .entity_swap import EntitySwapPerturbator
from .numeric import NumericPerturbator
from .negation import NegationTogglePerturbator
from .span_drop import SpanDropPerturbator
from .textattack_nli import TextAttackNLIFlipPerturbator
from ..nli import NLIVerifier
from ..config import SpanDropConfig, TextAttackConfig

DEFAULT_PERTURBATORS = [
    EntitySwapPerturbator(),
    NumericPerturbator(),
    NegationTogglePerturbator(),
]


def build_perturbators(
    *,
    enable_span_drop: bool = False,
    span_drop_min_words: int = SpanDropConfig.min_words,
    span_drop_max_words: int = SpanDropConfig.max_words,
    enable_textattack: bool = False,
    textattack_verifier: NLIVerifier | None = None,
    textattack_augmenter: str = TextAttackConfig.augmenter,
    textattack_pct_words_to_swap: float = TextAttackConfig.pct_words_to_swap,
    textattack_transformations_per_example: int = TextAttackConfig.transformations_per_example,
    textattack_search_calls: int = TextAttackConfig.search_calls,
    textattack_entail_threshold_neg: float = TextAttackConfig.entail_threshold_neg,
    textattack_contradiction_ratio: float = TextAttackConfig.contradiction_ratio,
    textattack_min_norm_edit: float = TextAttackConfig.min_norm_edit,
    textattack_max_norm_edit: float = TextAttackConfig.max_norm_edit,
    textattack_semantic_model_name: str | None = TextAttackConfig.semantic_model_name,
    textattack_min_semantic_similarity: float = TextAttackConfig.min_semantic_similarity,
    textattack_protect_entities: bool = TextAttackConfig.protect_entities,
    textattack_protect_numbers: bool = TextAttackConfig.protect_numbers,
    textattack_spacy_model: str = TextAttackConfig.spacy_model,
) -> list[Perturbator]:
    perturbators: list[Perturbator] = list(DEFAULT_PERTURBATORS)

    if enable_span_drop:
        perturbators.append(
            SpanDropPerturbator(
                min_words=span_drop_min_words,
                max_words=span_drop_max_words,
            )
        )

    if enable_textattack:
        if textattack_verifier is None:
            raise ValueError("textattack_verifier is required when enable_textattack=True.")
        perturbators.append(
            TextAttackNLIFlipPerturbator(
                verifier=textattack_verifier,
                augmenter=textattack_augmenter,
                pct_words_to_swap=textattack_pct_words_to_swap,
                transformations_per_example=textattack_transformations_per_example,
                search_calls=textattack_search_calls,
                entail_threshold_neg=textattack_entail_threshold_neg,
                contradiction_ratio=textattack_contradiction_ratio,
                min_norm_edit=textattack_min_norm_edit,
                max_norm_edit=textattack_max_norm_edit,
                semantic_model_name=textattack_semantic_model_name,
                min_semantic_similarity=textattack_min_semantic_similarity,
                protect_entities=textattack_protect_entities,
                protect_numbers=textattack_protect_numbers,
                spacy_model=textattack_spacy_model,
            )
        )
    return perturbators
