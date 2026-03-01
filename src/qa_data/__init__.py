"""Dataset adapters and builders for grounded QA."""

from .construct import ConstructionConfig, build_answerable_example
from .export import example_from_dict, example_to_dict
from .hotpot import HotpotSourceConfig, iter_hotpot_rows
from .negatives import NegativeSamplingConfig, derive_simple_unanswerable
from .records import ContextDocument, QaExample, SupportingSentence
from .split import SplitConfig, assign_split, assign_split_name

__all__ = [
    "ContextDocument",
    "QaExample",
    "SupportingSentence",
    "HotpotSourceConfig",
    "ConstructionConfig",
    "NegativeSamplingConfig",
    "SplitConfig",
    "iter_hotpot_rows",
    "build_answerable_example",
    "derive_simple_unanswerable",
    "assign_split",
    "assign_split_name",
    "example_to_dict",
    "example_from_dict",
]
