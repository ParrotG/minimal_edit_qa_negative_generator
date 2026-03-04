"""Dataset adapters and builders for grounded QA."""

from .construct import ConstructionConfig, build_answerable_example
from .export import example_from_dict, example_to_dict
from .hotpot import HotpotSourceConfig, iter_hotpot_rows
from .partition import partition_examples_by_data_split, write_partitioned_examples
from .records import ContextDocument, QaExample, SupportingSentence
from .tagging import (
    AnswerabilitySplitConfig,
    DataSplitConfig,
    assign_answerability_split_name,
    assign_data_split_name,
    iter_tagged_hotpot_rows,
    tag_hotpot_row,
)
from .unanswerable import (
    UnanswerableBuildConfig,
    build_examples_from_tagged_row,
    build_prepared_examples,
    build_unanswerable_from_answerable_example,
)

__all__ = [
    "AnswerabilitySplitConfig",
    "ConstructionConfig",
    "ContextDocument",
    "DataSplitConfig",
    "HotpotSourceConfig",
    "QaExample",
    "SupportingSentence",
    "UnanswerableBuildConfig",
    "assign_answerability_split_name",
    "assign_data_split_name",
    "build_answerable_example",
    "build_examples_from_tagged_row",
    "build_prepared_examples",
    "build_unanswerable_from_answerable_example",
    "example_from_dict",
    "example_to_dict",
    "iter_hotpot_rows",
    "iter_tagged_hotpot_rows",
    "partition_examples_by_data_split",
    "tag_hotpot_row",
    "write_partitioned_examples",
]
