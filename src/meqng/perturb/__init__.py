from .base import PerturbCandidate, Perturbator
from .entity_swap import EntitySwapPerturbator
from .numeric import NumericPerturbator
from .negation import NegationTogglePerturbator

DEFAULT_PERTURBATORS = [
    EntitySwapPerturbator(),
    NumericPerturbator(),
    NegationTogglePerturbator(),
]
