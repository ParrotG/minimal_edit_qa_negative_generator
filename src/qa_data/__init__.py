"""Dataset adapters and builders for grounded QA."""

from .construct import ConstructionConfig, build_answerable_example
from .export import example_from_dict, example_to_dict
from .hotpot import HotpotSourceConfig, iter_hotpot_rows
from .hotpot_raw import assign_hotpot_split, iter_split_hotpot_rows
from .negatives import NegativeSamplingConfig, derive_simple_unanswerable
from .records import ContextDocument, QaExample, SupportingSentence
from .split import SplitConfig, assign_split, assign_split_name
from .unanswerable import (
    UnanswerableBuildConfig,
    build_unanswerable_examples_from_pools,
    build_unanswerable_from_answerable_example,
    build_unanswerable_from_raw_hotpot_row,
    select_unanswerable_input_pools,
)

__all__ = [
    "ContextDocument",
    "QaExample",
    "SupportingSentence",
    "HotpotSourceConfig",
    "ConstructionConfig",
    "NegativeSamplingConfig",
    "UnanswerableBuildConfig",
    "SplitConfig",
    "iter_hotpot_rows",
    "iter_split_hotpot_rows",
    "build_answerable_example",
    "derive_simple_unanswerable",
    "build_unanswerable_examples_from_pools",
    "build_unanswerable_from_answerable_example",
    "build_unanswerable_from_raw_hotpot_row",
    "select_unanswerable_input_pools",
    "assign_split",
    "assign_hotpot_split",
    "assign_split_name",
    "example_to_dict",
    "example_from_dict",
]
